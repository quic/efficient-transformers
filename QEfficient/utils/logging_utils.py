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


class JSONNamespaceFormatter(logging.Formatter):
    """
    Custom formatter to output log records in JSON format with metadata.

    Methods:
        format(record): Formats a log record into a JSON string.

    Parameters:
        record (logging.LogRecord): The log record to format.

    Returns:
        str: JSON-formatted log string.
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
    Background thread to handle logging asynchronously using a queue.

    Attributes:
        logger (logging.Logger): Logger instance to handle log records.
        log_queue (queue.Queue): Queue from which log records are consumed.
        running (bool): Flag to control thread execution.
    """

    def __init__(self, logger, log_queue):
        """
        Initialize the logging thread.

        Parameters:
            logger (logging.Logger): Logger instance.
            log_queue (queue.Queue): Queue for log records.
        """
        super().__init__(daemon=True)
        self.logger = logger
        self.log_queue = log_queue
        self.running = True

    def run(self):
        """
        Continuously process log records from the queue and pass them to the logger.
        """
        while self.running:
            try:
                record = self.log_queue.get(timeout=1)
                self.logger.handle(record)
            except queue.Empty:
                continue

    def stop(self):
        """
        Stop the logging thread gracefully.
        """
        self.running = False


class QEFFLogger:
    """
    Singleton logger class for structured logging with namespace support.

    Class Attributes:
        _instance (Optional[logging.Logger]): Singleton logger instance.
        _logfile (Optional[str]): Path to the log file.
        _log_queue (queue.Queue): Queue for asynchronous logging.
        _logger_thread (Optional[QEFFLoggerThread]): Background logging thread.
    """

    _instance: Optional[logging.Logger] = None
    _logfile: Optional[str] = None
    _log_queue: queue.Queue = queue.Queue()
    _logger_thread: Optional[QEFFLoggerThread] = None

    def __init__(self, loglevel: Optional[str] = "INFO", log_path: Optional[str] = None):
        """
        Initialize the logger instance with specified log level and path.

        Parameters:
            loglevel (str): Logging level (e.g., "INFO", "DEBUG").
            log_path (str): Optional path to the log file.
        """
        if QEFFLogger._instance is None:
            self.loglevel = loglevel
            self.log_path = log_path
            self.logger = self._initialize_logger()
            QEFFLogger._instance = self.logger
            QEFFLogger._logger_thread = QEFFLoggerThread(self.logger, QEFFLogger._log_queue)
            QEFFLogger._logger_thread.start()

    def _initialize_logger(self) -> logging.Logger:
        """
        Set up the logger with rotating file handler and JSON formatter.

        Returns:
            logging.Logger: Configured logger instance.
        """
        if self.log_path is None:
            log_dir = os.path.expanduser("~/.cache/qefficient_logs")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(log_dir, f"QEFF_{timestamp}.log")

        QEFFLogger._logfile = self.log_path

        numeric_level = getattr(logging, self.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {self.loglevel}")

        logger = logging.getLogger("QEFF_LOGGER")
        logger.setLevel(numeric_level)

        if not logger.handlers:
            handler = RotatingFileHandler(self.log_path, maxBytes=5 * 1024 * 1024, backupCount=10)
            handler.setFormatter(JSONNamespaceFormatter())
            logger.addHandler(handler)

        return logger

    @classmethod
    def get_logger(
        cls, namespace: str, loglevel: Optional[str] = "INFO", log_path: Optional[str] = None
    ) -> logging.Logger:
        """
        Retrieve a logger adapter with a specific namespace.

        Parameters:
            namespace (str): Logical grouping for the log.
            loglevel (str): Logging level.
            log_path (str): Optional path to the log file.

        Returns:
            logging.Logger: Logger adapter with namespace.
        """
        if cls._instance is None:
            cls(loglevel, log_path)
        return logging.LoggerAdapter(cls._instance, {"namespace": namespace})

    @classmethod
    def log(cls, level: str, namespace: str, msg: str, fn: str = "", lno: int = 0, func: str = ""):
        """
        Log a message with specified level and metadata.

        Parameters:
            level (str): Logging level (e.g., "INFO", "ERROR").
            namespace (str): Logical grouping for the log.
            msg (str): Log message.
            fn (str): Filename where the log is generated.
            lno (int): Line number in the file.
            func (str): Function name.
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
    def set_loglevel(cls, loglevel: Optional[str] = "INFO"):
        """
        Update the log level of the logger.

        Parameters:
            loglevel (str): New log level to set.
        """
        if cls._instance is None:
            raise RuntimeError("Logger has not been initialized yet. Call get_logger() first.")

        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {loglevel}")

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
        Parse the line-delimited JSON log in cls._logfile and print timing table with t1 as baseline (0.0s):
          - Model Loading      : t2 - t1
          - Model Exporting    : t3 - t2
          - Model Compilation  : t4 - t3
          - Text Generation    : t5 - t4
          - Total Time         : t5 - t1

        Milestones (matched to your log sample):
          t1: first log line timestamp (baseline)
          t2: "PyTorch export successful"
          t3: "Transformed ONNX saved"
          t4: "Model compilation is finished and saved"
          t5: "Text Generated finised"
             If t5 is missing, we fall back to "specialization_file_path" as readiness marker.
        """
        path = cls._logfile
        if not path:
            raise FileNotFoundError("Log file path is not set (cls._logfile is None).")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Log file does not exist: {path}")

        t_start: Optional[datetime] = None
        t_export_done: Optional[datetime] = None
        t_onnx_saved: Optional[datetime] = None
        t_compile_done: Optional[datetime] = None
        t_text_done: Optional[datetime] = None
        t_text_ready: Optional[datetime] = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
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

                if t_start is None:
                    t_start = ts

                if ("PyTorch export successful" in msg) and (t_export_done is None):
                    t_export_done = ts

                if ("Transformed ONNX saved" in msg) and (t_onnx_saved is None):
                    t_onnx_saved = ts

                if ("Model compilation is finished and saved" in msg) and (t_compile_done is None):
                    t_compile_done = ts

                if ("Text Generated finised" in msg) and (t_text_done is None):
                    t_text_done = ts

                if ("specialization_file_path" in msg) and (t_text_ready is None):
                    t_text_ready = ts

        if t_start is None:
            raise ValueError("Could not determine start time (no valid log lines with date/time).")

        if t_text_done is None:
            t_text_done = t_text_ready

        t_export_done = t_export_done or t_start
        t_onnx_saved = t_onnx_saved or t_export_done
        t_compile_done = t_compile_done or t_onnx_saved
        t_text_done = t_text_done or t_compile_done

        def to_offset_seconds(t: datetime) -> float:
            return (t - t_start).total_seconds()

        o1 = 0.0
        o2 = to_offset_seconds(t_export_done)
        o3 = to_offset_seconds(t_onnx_saved)
        o4 = to_offset_seconds(t_compile_done)
        o5 = to_offset_seconds(t_text_done)

        timing_data: List[List[Any]] = [
            ["Model Loading", max(0.0, o2 - o1)],
            ["Model Exporting", max(0.0, o3 - o2)],
            ["Model Compilation", max(0.0, o4 - o3)],
            ["Text Generation", max(0.0, o5 - o4)],
            ["Total Time", max(0.0, o5 - o1)],
        ]

        print(tabulate(timing_data, headers=["Step", "Time (s)"], tablefmt="github", floatfmt=".3f"))
