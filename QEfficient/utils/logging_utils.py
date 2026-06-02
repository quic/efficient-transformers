# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import logging
import os
import threading
from datetime import datetime
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
        return json.dumps(log_record)


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

    @classmethod
    def _resolve_log_path(cls, requested_path: Optional[str]) -> str:
        """Resolve the final log file path from a user path or defaults."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_file = os.path.join(LoggerConfig.default_log_dir, f"QEFF_{timestamp}.log")
        if not requested_path:
            os.makedirs(LoggerConfig.default_log_dir, exist_ok=True)
            return default_file

        path = Path(requested_path).expanduser()
        if path.suffix.lower() == ".log":
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)

        path.mkdir(parents=True, exist_ok=True)
        return str(path / f"QEFF_{timestamp}.log")

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
        )
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
        if cls._instance:
            handlers = cls._instance.handlers[:]
            for handler in handlers:
                handler.flush()
                handler.close()
                cls._instance.removeHandler(handler)
            cls._instance = None
            cls._logfile = None

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
    def _extract_milestone_times(cls, path: str) -> Dict[str, datetime]:
        """
        Extract first occurrence timestamp for each milestone key from JSON log lines.
        """
        milestone_patterns: Dict[str, Tuple[str, ...]] = {
            "START_LOAD": ("initiating the model weight loading",),
            "LOAD_DONE": ("pytorch transforms applied to model",),
            "ONNX_SAVED": ("model export is finished and saved", "transformed onnx saved"),
            "COMPILE_DONE": ("model compilation is finished and saved",),
            "TEXT_DONE": ("text generation finished", "text generated finised"),
            "TEXT_READY": ("specialization_file_path",),
        }

        times: Dict[str, datetime] = {}
        for record in cls._iter_log_records(path):
            message = str(record.get("message", "")).lower()
            timestamp = cls._get_record_timestamp(record)
            if not timestamp:
                continue

            for key, patterns in milestone_patterns.items():
                if key in times:
                    continue
                if any(pattern in message for pattern in patterns):
                    times[key] = timestamp
                    break
        return times

    @classmethod
    def print_table(cls) -> bool:
        """
        Parse the line-delimited JSON log in cls._logfile and print timing table with t1 as baseline (0.0s).
        """
        path = cls._logfile
        if not path:
            return False
        if not os.path.exists(path):
            return False

        times = cls._extract_milestone_times(path)
        if not times:
            return False

        t_start = times.get("START_LOAD", min(times.values()))
        t2 = times.get("LOAD_DONE", t_start)  # end of loading
        t3 = times.get("ONNX_SAVED", t2)  # export end
        t4 = times.get("COMPILE_DONE", t3)  # compile end
        t5 = times.get("TEXT_DONE", times.get("TEXT_READY", t4))  # text gen end

        # Keep boundaries monotonic for stable table output.
        if t2 < t_start:
            t2 = t_start
        if t3 < t2:
            t3 = t2
        if t4 < t3:
            t4 = t3
        if t5 < t4:
            t5 = t4

        def offset_seconds(t: datetime) -> float:
            return (t - t_start).total_seconds()

        o1 = 0.0
        o2 = offset_seconds(t2)
        o3 = offset_seconds(t3)
        o4 = offset_seconds(t4)
        o5 = offset_seconds(t5)

        timing_data: List[List[Any]] = [
            ["Model Loading", max(0.0, o2 - o1)],
            ["Model Exporting", max(0.0, o3 - o2)],
            ["Model Compilation", max(0.0, o4 - o3)],
            ["Text Generation", max(0.0, o5 - o4)],
            ["Total Time", max(0.0, o5 - o1)],
        ]
        print("\n")
        print(tabulate(timing_data, headers=["Step", "Time (s)"], tablefmt="github", floatfmt=".3f"))
        return True
