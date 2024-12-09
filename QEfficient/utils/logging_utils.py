# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import csv
import logging
import os
import threading
import time

import psutil


class QEffFormatter(logging.Formatter):
    """
    Formatter class used to set colors for printing different logging levels of messages on console.
    """

    cyan: str = "\x1b[38;5;14m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    common_format: str = "%(levelname)s - %(name)s - %(message)s"  # type: ignore
    format_with_line_info = "%(levelname)s - %(name)s - %(message)s  (%(filename)s:%(lineno)d)"  # type: ignore

    FORMATS = {
        logging.DEBUG: cyan + format_with_line_info + reset,
        logging.INFO: cyan + common_format + reset,
        logging.WARNING: yellow + common_format + reset,
        logging.ERROR: red + format_with_line_info + reset,
        logging.CRITICAL: bold_red + format_with_line_info + reset,
    }

    def format(self, record):
        """
        Overriding the base class method to Choose format based on log level.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def create_logger() -> logging.Logger:
    """
    Creates a logger object with Colored QEffFormatter.
    """
    logger = logging.getLogger("QEfficient")

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # define formatter
    ch.setFormatter(QEffFormatter())

    logger.addHandler(ch)
    return logger


# Define the logger object that can be used for logging purposes throughout the module.
logger = create_logger()


class MemoryTracker:
    def __init__(self, interval=1):
        self.interval = interval
        self.tracking = False
        self.max_mem_usage = 0
        self.duration = 0.0
        self.thread = None

    def start(self, init_mem=0):
        self.tracking = True
        self.max_mem_usage = init_mem
        self.duration = time.perf_counter()
        self.thread = threading.Thread(target=self.track_memory)
        self.thread.start()

    def stop(self):
        self.tracking = False
        self.thread.join()
        self.duration = (time.perf_counter() - self.duration) // 1

    def track_memory(self):
        process = psutil.Process(os.getpid())
        while self.tracking:
            mem_info = process.memory_info()
            curr_mem_usage = mem_info.rss // (1024 * 1024)
            self.max_mem_usage = max(self.max_mem_usage, curr_mem_usage)
            time.sleep(self.interval)


def tabulate_measurements(fields, file):
    if not os.path.exists(file):
        with open(file, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(list(fields.keys()))
    with open(file, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(list(fields.values()))
