# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import atexit
import inspect
import json
import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tabulate import tabulate

# Import centralized config
from QEfficient.utils.constants import LoggerConfig


class JSONNamespaceFormatter(logging.Formatter):
    """
    Custom formatter to output log records in JSON format with metadata.
    """

    def format(self, record):
        log_record = {
            "created": record.created,
            "date": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d"),
            "time": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
            "level": record.levelname,
            "namespace": getattr(record, "namespace", "default"),
            "file": record.filename,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        for field in ("event", "api", "arguments"):
            value = getattr(record, field, None)
            if value is not None:
                log_record[field] = value
        for field in ("run_id", "model", "milestone", "status", "error"):
            value = getattr(record, field, None)
            if value is not None:
                log_record[field] = value
        return json.dumps(log_record)


_SENSITIVE_ARGUMENT_NAMES = {
    "api_key",
    "authorization",
    "access_token",
    "hf_token",
    "password",
    "secret",
    "token",
    "use_auth_token",
}


def _serialize_argument(value: Any, name: Optional[str] = None) -> Any:
    """Return a compact JSON-safe representation of an API argument."""
    if name and name.lower() in _SENSITIVE_ARGUMENT_NAMES:
        return "<redacted>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize_argument(item, str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_argument(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        result = {"type": type(value).__name__, "shape": list(value.shape), "dtype": str(value.dtype)}
        if hasattr(value, "device"):
            result["device"] = str(value.device)
        return result
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return {
                "type": f"{type(value).__module__}.{type(value).__name__}",
                "config": _serialize_argument(value.to_dict()),
            }
        except Exception:
            pass
    return {
        "type": f"{type(value).__module__}.{type(value).__name__}",
        "repr": " ".join(str(value).split())[:200],
    }


def log_api_arguments(api: str, namespace: str, arguments: Dict[str, Any]) -> None:
    """Write one structured API-argument record."""
    QEFFLogger.log_event(
        "api_call",
        namespace,
        "Captured API arguments.",
        api=api,
        arguments={key: _serialize_argument(value, key) for key, value in arguments.items()},
    )


def log_generate_call(func):
    """Log generate arguments and a plain completion message."""
    signature = inspect.signature(func)
    receiver_name = next(iter(signature.parameters), None)

    @wraps(func)
    def wrapper(*args, **kwargs):
        receiver = args[0]
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop(receiver_name, None)
        namespace = type(receiver).__name__
        log_api_arguments("generate", namespace, arguments)
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            QEFFLogger.log_api_failure("generate", namespace, exc)
            raise
        QEFFLogger.log_event(
            "milestone",
            namespace,
            "Generation completed.",
            api="generate",
            milestone="generation_complete",
        )
        return result

    wrapper._qeff_api_dump = True
    return wrapper


def log_pipeline_api(
    api: str,
    completion_message: str,
    completion_milestone: str,
    start_run: bool = False,
    finish_run: bool = False,
):
    """Log one public Diffusers API and optionally manage its timing run.

    ``from_pretrained``, ``export``, and ``compile`` suppress duplicate API
    records emitted by child models.  A pipeline ``__call__`` can use the same
    decorator with ``start_run=True`` and ``finish_run=True`` to log generate
    arguments and close the timing run.
    """

    def decorator(func):
        signature = inspect.signature(func)
        receiver_name = next(iter(signature.parameters), None)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            receiver = arguments.pop(receiver_name, None)
            namespace = receiver.__name__ if isinstance(receiver, type) else type(receiver).__name__
            if api == "from_pretrained":
                QEFFLogger.start_run(str(arguments.get("pretrained_model_name_or_path", "unknown")))
            elif start_run and not QEFFLogger._run_active:
                QEFFLogger.start_run(namespace)

            log_api_arguments(api, namespace, arguments)
            try:
                if api in {"from_pretrained", "export", "compile"}:
                    with QEFFLogger.suppress_api_logging():
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            except Exception as exc:
                QEFFLogger.log_api_failure(api, namespace, exc)
                raise
            else:
                QEFFLogger.log_event(
                    "milestone",
                    namespace,
                    completion_message,
                    api=api,
                    milestone=completion_milestone,
                )
                return result
            finally:
                if finish_run:
                    QEFFLogger.finish_run()

        wrapper._qeff_api_dump = True
        return wrapper

    return decorator


def log_from_pretrained_call(func):
    """Start a model run and capture ``from_pretrained`` arguments before loading."""
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        bound = signature.bind_partial(cls, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("cls", None)
        model_name = arguments.get("pretrained_model_name_or_path", "unknown")
        if QEFFLogger.is_api_logging_suppressed():
            return func(cls, *args, **kwargs)
        QEFFLogger.start_run(str(model_name))
        log_api_arguments("from_pretrained", cls.__name__, arguments)
        try:
            result = func(cls, *args, **kwargs)
        except Exception as exc:
            QEFFLogger.log_api_failure("from_pretrained", cls.__name__, exc)
            raise
        QEFFLogger.log_event(
            "milestone",
            cls.__name__,
            "Model loading completed.",
            api="from_pretrained",
            milestone="load_complete",
        )
        return result

    wrapper._qeff_api_dump = True
    return wrapper


class QEFFLogger:
    """
    Singleton logger class for structured logging with namespace support.

    Project-wide behavior:
      - A single log level is enforced using env `QEFF_LOG_LEVEL` (default = INFO).
      - Log path resolved with priority: explicit arg > env `QEFF_LOG_PATH` > default dir + timestamp.
    """

    _instance: Optional[logging.Logger] = None
    _logfile: Optional[str] = None
    _init_lock = threading.Lock()
    _table_written = False
    _summary_printed = False
    _atexit_registered = False
    _run_active = False
    _run_counter = 0
    _current_run_id: Optional[int] = None
    _current_model: Optional[str] = None
    _api_logging_suppressed = 0
    _run_owner_thread: Optional[int] = None
    _run_lock = threading.RLock()

    def __init__(self, loglevel: Optional[str] = None, log_path: Optional[str] = None):
        """
        Initialize the logger instance with specified path. Level is globally controlled by env.
        Args:
            loglevel: kept for backward compatibility, but env `QEFF_LOG_LEVEL` takes precedence.
            log_path: optional path to the log file (highest priority).
        """
        with QEFFLogger._init_lock:
            if QEFFLogger._instance is not None:
                return

            # Determine effective log level:
            # Priority: ENV(QEFF_LOG_LEVEL) -> arg(loglevel) -> LoggerConfig.default_level
            env_level = os.environ.get(LoggerConfig.log_level_env)
            effective_level_name = (env_level or loglevel or LoggerConfig.default_level).upper()
            numeric_level = getattr(logging, effective_level_name, None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {effective_level_name}")
            self.loglevel = effective_level_name

            # Resolve log path (arg > env > default dir + timestamp)
            env_path = os.environ.get(LoggerConfig.log_path_env)
            self.log_path = self._resolve_log_path(log_path or env_path)

            # Initialize the base logger
            self.logger = self._initialize_logger()
            QEFFLogger._instance = self.logger
            QEFFLogger._table_written = False
            QEFFLogger._summary_printed = False
            if not QEFFLogger._atexit_registered:
                atexit.register(QEFFLogger._finalize)
                QEFFLogger._atexit_registered = True

    @classmethod
    def _resolve_log_path(cls, requested_path: Optional[str]) -> str:
        """Resolve the final log file path from a user path or defaults."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        process_suffix = f"_{os.getpid()}"
        default_file = os.path.join(LoggerConfig.default_log_dir, f"QEFF_{timestamp}{process_suffix}.log")
        if not requested_path:
            os.makedirs(LoggerConfig.default_log_dir, exist_ok=True)
            return default_file

        path = Path(requested_path).expanduser()
        if path.suffix.lower() == ".log":
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)

        path.mkdir(parents=True, exist_ok=True)
        return str(path / f"QEFF_{timestamp}{process_suffix}.log")

    def _initialize_logger(self) -> logging.Logger:
        """
        Set up the logger with rotating file handler and JSON formatter.
        """
        QEFFLogger._logfile = self.log_path

        logger = logging.getLogger("QEFF_LOGGER")
        logger.setLevel(getattr(logging, self.loglevel))
        logger.propagate = False

        # Avoid duplicate handlers if reinitialized in same process
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        handler = RotatingFileHandler(
            self.log_path,
            maxBytes=LoggerConfig.max_bytes,
            backupCount=LoggerConfig.backup_count,
            delay=True,
        )

        class RunContextFilter(logging.Filter):
            def filter(self, record):
                if QEFFLogger._run_active:
                    record.run_id = QEFFLogger._current_run_id
                    record.model = QEFFLogger._current_model
                return True

        handler.addFilter(RunContextFilter())
        handler.setFormatter(JSONNamespaceFormatter())
        logger.addHandler(handler)

        return logger

    @classmethod
    def get_logger(
        cls, namespace: str, loglevel: Optional[str] = None, log_path: Optional[str] = None
    ) -> logging.Logger:
        """
        Retrieve a logger adapter with a specific namespace.
        Note: project-wide level comes from env `QEFF_LOG_LEVEL` (default INFO).
        """
        if cls._instance is None:
            cls(loglevel, log_path)
        return logging.LoggerAdapter(cls._instance, {"namespace": namespace})

    @classmethod
    def log_event(cls, event: str, namespace: str, message: str, **fields: Any) -> None:
        """Write one structured event record."""
        if cls._api_logging_suppressed and event in {"api_call", "milestone", "api_failure"}:
            return
        if cls._instance is None:
            cls.get_logger(namespace)
        extra = {"namespace": namespace, "event": event, **fields}
        logging.LoggerAdapter(cls._instance, extra).info(message, stacklevel=2)

    @classmethod
    def log_api_failure(cls, api: str, namespace: str, error: Exception) -> None:
        cls.log_event(
            "api_failure",
            namespace,
            f"{api} failed.",
            api=api,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )

    @classmethod
    @contextmanager
    def suppress_api_logging(cls):
        cls._api_logging_suppressed += 1
        try:
            yield
        finally:
            cls._api_logging_suppressed -= 1

    @classmethod
    def is_api_logging_suppressed(cls) -> bool:
        return cls._api_logging_suppressed > 0

    @classmethod
    def log(cls, level: str, namespace: str, msg: str, fn: str = "", lno: int = 0, func: str = ""):
        """
        Log a message with specified level and metadata.
        """
        if cls._instance is None:
            raise RuntimeError("Logger has not been initialized. Call get_logger() first.")

        level_num = getattr(logging, level.upper(), None)
        if not isinstance(level_num, int):
            raise ValueError(f"Invalid log level: {level}")

        logger = logging.LoggerAdapter(cls._instance, {"namespace": namespace})
        logger.log(level_num, msg, stacklevel=2)

    @classmethod
    def set_loglevel(cls, loglevel: Optional[str] = None):
        """
        Update the log level of the logger at runtime.
        Priority remains ENV > arg > default.
        If ENV is set, it will continue to override; otherwise arg/default apply.
        """
        if cls._instance is None:
            raise RuntimeError("Logger has not been initialized yet. Call get_logger() first.")

        env_level = os.environ.get(LoggerConfig.log_level_env)
        effective_level_name = (env_level or loglevel or LoggerConfig.default_level).upper()
        numeric_level = getattr(logging, effective_level_name, None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {effective_level_name}")

        cls._instance.setLevel(numeric_level)

    @classmethod
    def close_logger(cls):
        """
        Gracefully shut down the logger.
        """
        with cls._run_lock:
            if cls._instance:
                cls._finalize()
                cls._close_handlers()
                cls._run_active = False
                cls._current_run_id = None
                cls._current_model = None
                cls._run_owner_thread = None

    @classmethod
    def _close_handlers(cls) -> None:
        if cls._instance is None:
            return
        for handler in cls._instance.handlers[:]:
            handler.flush()
            handler.close()
            cls._instance.removeHandler(handler)
        cls._instance = None
        cls._logfile = None

    @classmethod
    def start_run(cls, model_name: str) -> None:
        """Start a model run inside the current process log file."""
        with cls._run_lock:
            if cls._instance is None:
                cls()
            current_thread = threading.get_ident()
            if cls._run_active and cls._run_owner_thread not in (None, current_thread):
                raise RuntimeError("QEFFLogger supports one active model run per process.")
            if cls._run_active:
                cls._append_final_table()

            cls._run_counter += 1
            cls._current_run_id = cls._run_counter
            cls._current_model = model_name
            cls._run_active = True
            cls._run_owner_thread = current_thread
            cls._table_written = False
            cls._summary_printed = False
            cls.log_event(
                "milestone",
                "MODEL",
                f"Starting model weight loading: {model_name}.",
                milestone="load_start",
            )

    @classmethod
    def finish_run(cls) -> None:
        """Write the current run's timing table and release its run state."""
        if cls._instance is None or not cls._run_active:
            return
        if cls._append_final_table():
            cls._run_active = False
            cls._run_owner_thread = None

    @classmethod
    def _build_timing_table(cls, run_id: Optional[int] = None) -> Optional[str]:
        """Build the timing table for one model run from the process log."""
        path = cls._logfile
        if not path or not os.path.exists(path):
            return None

        times = cls._extract_milestone_times(path, run_id=run_id)
        required_milestones = ("START_LOAD", "LOAD_DONE")
        if any(key not in times for key in required_milestones):
            return None
        if "COMPILE_DONE" not in times and "COMPILE_FAIL" not in times:
            return None

        records = [record for record in cls._iter_log_records(path) if run_id is None or record.get("run_id") == run_id]
        messages = [str(record.get("message", "")).lower() for record in records]
        milestones = {record.get("milestone") for record in records}
        export_completed = any(milestone in milestones for milestone in {"export_complete"}) or any(
            "onnx export completed" in message
            or "transformed onnx saved" in message
            or "model export is finished and saved" in message
            for message in messages
        )
        compile_completed = "compile_complete" in milestones or any(
            "compilation completed" in message and "cached qpc" not in message for message in messages
        )
        export_skipped = not export_completed and (
            "export_skipped" in milestones or any("onnx export skipped" in message for message in messages)
        )
        compile_skipped = not compile_completed and (
            "compile_skipped" in milestones
            or any(
                "compilation skipped" in message or "compilation completed (cached qpc)" in message
                for message in messages
            )
        )

        t_start = times.get("START_LOAD", min(times.values()))
        t_load_done = max(times.get("LOAD_DONE", t_start), t_start)
        t_export_done = max(times.get("ONNX_SAVED", t_load_done), t_load_done)
        t_compile_done = max(
            times.get("COMPILE_DONE", times.get("COMPILE_FAIL", t_export_done)),
            t_export_done,
        )

        loading = max(0.0, (t_load_done - t_start).total_seconds())
        exporting = 0.0 if export_skipped else max(0.0, (t_export_done - t_load_done).total_seconds())
        compiling = 0.0 if compile_skipped else max(0.0, (t_compile_done - t_export_done).total_seconds())

        if times.get("GENERATE_START") and times.get("TEXT_DONE"):
            # Diffusers pipelines log the public __call__ API before their
            # internal export/compile work. Do not count that setup time as
            # generation or double-count it in the total.
            generation_start = max(times["GENERATE_START"], t_compile_done)
            generation = max(0.0, (times["TEXT_DONE"] - generation_start).total_seconds())
        elif times.get("TEXT_DONE"):
            generation = max(0.0, (times["TEXT_DONE"] - t_compile_done).total_seconds())
        else:
            generation = 0.0

        total = loading + exporting + compiling + generation
        timing_data: List[List[Any]] = [
            ["Model Loading", loading],
            ["Model Exporting", exporting],
            ["Model Compilation", compiling],
            ["Text Generation", generation],
            ["Total Time", total],
        ]
        return tabulate(timing_data, headers=["Step", "Time (s)"], tablefmt="github", floatfmt=".3f")

    @classmethod
    def _append_final_table(cls, table: Optional[str] = None) -> bool:
        """Append the current model's timing table to the shared process log."""
        if cls._table_written or cls._instance is None:
            return False
        table = table or cls._build_timing_table(cls._current_run_id)
        if table is None or cls._logfile is None:
            return False

        for handler in cls._instance.handlers:
            handler.flush()
        with open(cls._logfile, "a", encoding="utf-8") as handle:
            model_label = cls._current_model or "process"
            # The JSON handler already leaves the cursor after a newline. Keep
            # exactly one blank line before and after the human-readable table.
            handle.write(f"\n===== QEfficient Timing Summary: {model_label} =====\n")
            handle.write(table)
            handle.write("\n\n")
        cls._table_written = True
        return True

    @classmethod
    def _finalize(cls) -> None:
        """Write the final table and print the run summary once."""
        if cls._instance is None or cls._summary_printed:
            return
        cls._append_final_table()
        if cls._logfile:
            print(f"Log file: {cls._logfile}")
        cls._summary_printed = True

    @classmethod
    def _parse_dt(cls, date_str: str, time_str: str) -> datetime:
        """Parse 'YYYY-MM-DD' and 'HH:MM:SS' into a datetime."""
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

    @classmethod
    def get_logfile_path(cls) -> Optional[str]:
        """Return active log file path, if logger is initialized."""
        return cls._logfile

    @classmethod
    def _iter_log_records(cls, path: str) -> Iterable[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record

    @classmethod
    def _records_for_run(cls, path: str, run_id: Optional[int]) -> List[Dict[str, Any]]:
        """Return run records plus unscoped setup records immediately before the run."""
        records = list(cls._iter_log_records(path))
        if run_id is None:
            return records

        run_records = [record for record in records if record.get("run_id") == run_id]
        run_timestamps = [cls._get_record_timestamp(record) for record in run_records]
        run_timestamps = [timestamp for timestamp in run_timestamps if timestamp is not None]
        if not run_timestamps:
            return run_records

        run_start = min(run_timestamps)
        return [
            record
            for record in records
            if record.get("run_id") == run_id
            or (
                record.get("run_id") is None
                and (timestamp := cls._get_record_timestamp(record)) is not None
                and timestamp <= run_start
            )
        ]

    @classmethod
    def _get_record_timestamp(cls, record: Dict[str, Any]) -> Optional[datetime]:
        created = record.get("created")
        if isinstance(created, (float, int)):
            return datetime.fromtimestamp(float(created))

        date_str = record.get("date")
        time_str = record.get("time")
        if not date_str or not time_str:
            return None
        try:
            return cls._parse_dt(str(date_str), str(time_str))
        except ValueError:
            return None

    @classmethod
    def _extract_milestone_times(cls, path: str, run_id: Optional[int] = None) -> Dict[str, datetime]:
        """
        Extract first occurrence timestamp for each milestone key from JSON log lines.
        """
        milestone_patterns: Dict[str, Tuple[str, ...]] = {
            "START_LOAD": ("starting model weight loading", "initiating the model weight loading"),
            "LOAD_DONE": (
                "applied pytorch transforms to model",
                "pytorch transforms applied to model",
            ),
            "ONNX_SAVED": (
                "model export is finished and saved",
                "transformed onnx saved",
                "onnx export completed",
                "onnx export skipped",
            ),
            "COMPILE_DONE": (
                "model compilation is finished and saved",
                "compilation completed",
                "compilation skipped",
            ),
            "TEXT_DONE": (
                "text generation finished",
                "generation completed",
            ),
        }

        times: Dict[str, datetime] = {}
        structured_milestones = {
            "load_start": "START_LOAD",
            "load_complete": "LOAD_DONE",
            "export_complete": "ONNX_SAVED",
            "export_skipped": "ONNX_SAVED",
            "compile_complete": "COMPILE_DONE",
            "compile_skipped": "COMPILE_DONE",
            "generation_complete": "TEXT_DONE",
        }
        for record in cls._records_for_run(path, run_id):
            message = str(record.get("message", "")).lower()
            timestamp = cls._get_record_timestamp(record)
            if not timestamp:
                continue

            if record.get("event") == "api_call" and record.get("api") == "generate":
                times.setdefault("GENERATE_START", timestamp)

            milestone = structured_milestones.get(record.get("milestone"))
            if milestone:
                if milestone == "START_LOAD":
                    times.setdefault(milestone, timestamp)
                else:
                    times[milestone] = timestamp
                continue

            if record.get("event") == "api_failure" and record.get("api") == "compile":
                times["COMPILE_FAIL"] = timestamp
                continue

            for key, patterns in milestone_patterns.items():
                if run_id is not None and record.get("run_id") is None and key != "LOAD_DONE":
                    continue
                if key in times and key not in {"ONNX_SAVED", "COMPILE_DONE", "TEXT_DONE"}:
                    continue
                if any(pattern in message for pattern in patterns):
                    times[key] = timestamp
        return times

    @classmethod
    def print_table(cls) -> bool:
        """
        Append and print the timing table with t1 as baseline (0.0s).
        """
        table = cls._build_timing_table()
        if table is None:
            return False
        cls._append_final_table(table)
        print("\n")
        print(table)
        return True
