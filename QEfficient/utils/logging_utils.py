# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import logging
import os
import queue
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from tabulate import tabulate

# Import centralized config
from QEfficient.utils.constants import LoggerConfig


class JSONNamespaceFormatter(logging.Formatter):
    """
    Custom formatter to output log records in JSON format with metadata.
    """

    def format(self, record):
        log_record = {
            "date": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d"),
            "time": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
            "level": record.levelname,
            "namespace": getattr(record, "namespace", "default"),
            "file": record.filename,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


class QEFFLoggerThread(threading.Thread):
    """
    Custom formatter to output log records in JSON format with metadata.

    Methods:
        format(record): Formats a log record into a JSON string.

    Parameters:
        record (logging.LogRecord): The log record to format.

    Returns:
        str: JSON-formatted log string.
    """

    def __init__(self, logger, log_queue):
        super().__init__(daemon=True)
        self.logger = logger
        self.log_queue = log_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                record = self.log_queue.get(timeout=1)
                self.logger.handle(record)
            except queue.Empty:
                continue

    def stop(self):
        self.running = False


class QEFFLogger:
    """
    Singleton logger class for structured logging with namespace support.

    Project-wide behavior:
      - A single log level is enforced using env `QEFF_LOG_LEVEL` (default = INFO).
      - Log path resolved with priority: explicit arg > env `QEFF_LOG_PATH` > default dir + timestamp.
    """

    _instance: Optional[logging.Logger] = None
    _logfile: Optional[str] = None
    _log_queue: queue.Queue = queue.Queue()
    _logger_thread: Optional[QEFFLoggerThread] = None

    def __init__(self, loglevel: Optional[str] = None, log_path: Optional[str] = None):
        """
        Initialize the logger instance with specified path. Level is globally controlled by env.
        Args:
            loglevel: kept for backward compatibility, but env `QEFF_LOG_LEVEL` takes precedence.
            log_path: optional path to the log file (highest priority).
        """
        if QEFFLogger._instance is None:
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
            self.log_path = log_path or env_path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not self.log_path:
                os.makedirs(LoggerConfig.default_log_dir, exist_ok=True)
                self.log_path = os.path.join(LoggerConfig.default_log_dir, f"QEFF_{timestamp}.log")
            else:
                os.makedirs(self.log_path, exist_ok=True)
                self.log_path = os.path.join(self.log_path, f"QEFF_{timestamp}.log")
            # Initialize the base logger and start background thread
            self.logger = self._initialize_logger()
            QEFFLogger._instance = self.logger
            QEFFLogger._logger_thread = QEFFLoggerThread(self.logger, QEFFLogger._log_queue)
            QEFFLogger._logger_thread.start()

    def _initialize_logger(self) -> logging.Logger:
        """
        Set up the logger with rotating file handler and JSON formatter.
        """
        QEFFLogger._logfile = self.log_path

        logger = logging.getLogger("QEFF_LOGGER")
        logger.setLevel(getattr(logging, self.loglevel))

        # Avoid duplicate handlers if reinitialized in same process
        if not logger.handlers:
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

        record = cls._instance.makeRecord(
            name="QEFF_LOGGER",
            level=level_num,
            fn=fn,
            lno=lno,
            msg=msg,
            args=(),
            exc_info=None,
            func=func,
            extra={"namespace": namespace},
        )
        cls._log_queue.put(record)

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
        Gracefully shut down the logger and its thread.
        """
        if cls._logger_thread:
            cls._logger_thread.stop()
            cls._logger_thread.join()
            cls._logger_thread = None

        if cls._instance:
            handlers = cls._instance.handlers[:]
            for handler in handlers:
                handler.close()
                cls._instance.removeHandler(handler)
            cls._instance = None
            cls._logfile = None

    @classmethod
    def _parse_dt(cls, date_str: str, time_str: str) -> datetime:
        """Parse 'YYYY-MM-DD' and 'HH:MM:SS' into a datetime."""
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

    @classmethod
    def print_table(cls) -> None:
        """
        Parse the line-delimited JSON log in cls._logfile and print timing table with t1 as baseline (0.0s).
        """
        path = cls._logfile
        if not path:
            raise FileNotFoundError(f"Log file path is not set ({cls._logfile} is None).")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Log file does not exist: {path}")

        SUBSTR_TO_KEY: Dict[str, str] = {
            "initiating the model weight loading.": "START_LOAD",
            "pytorch transforms applied to model": "LOAD_DONE",
            "transformed onnx saved": "ONNX_SAVED",
            "model compilation is finished and saved": "COMPILE_DONE",
            "text generated finised": "TEXT_DONE",
            "specialization_file_path": "TEXT_READY",
        }

        def classify(msg: str) -> Optional[str]:
            m = msg.lower()
            for needle, key in SUBSTR_TO_KEY.items():
                if needle in m:
                    return key
            return None

        from datetime import timedelta

        t_start: Optional[datetime] = None
        last_ts: Optional[datetime] = None
        times: Dict[str, datetime] = {}

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue

                date_str = rec.get("date")
                time_str = rec.get("time")
                msg = rec.get("message", "")
                if not date_str or not time_str:
                    continue

                ts = cls._parse_dt(date_str, time_str)

                # Enforce strictly increasing timestamps
                if last_ts is not None and ts <= last_ts:
                    ts = last_ts + timedelta(milliseconds=1)

                key = classify(msg)
                if key and key not in times:
                    times[key] = ts
                    if key == "START_LOAD" and t_start is None:
                        t_start = ts

                last_ts = ts

        if t_start is None:
            logging.warning(
                "Missing required milestone: 'Initiating the model weight loading.' "
                "Defaulting t_start to first available timestamp (0.0 baseline)."
            )

            if times:
                # Use earliest recorded milestone as baseline
                t_start = min(times.values())
            else:
                # Absolute fallback: zero baseline
                t_start = datetime.min

        t2 = times.get("LOAD_DONE", t_start)  # end of loading
        t3 = times.get("ONNX_SAVED") or t2  # export end
        t4 = times.get("COMPILE_DONE") or t3  # compile end
        t5 = times.get("TEXT_DONE") or times.get("TEXT_READY") or t4  # text gen end

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
