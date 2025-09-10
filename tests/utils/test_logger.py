# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import threading
import time

from QEfficient.utils.logging_utils import QEFFLogger

# -------------------------------
# Define namespace once
# -------------------------------
NAMESPACE = "model"

# -------------------------------
# Initialize logger
# -------------------------------
logger = QEFFLogger.get_logger(NAMESPACE, "DEBUG")


# -------------------------------
# Worker function for threads
# -------------------------------
def log_worker(thread_id):
    for i in range(5):
        logger.info(f"Thread-{thread_id} logging message {i}")
        time.sleep(0.1)


# -------------------------------
# Create and start threads
# -------------------------------
threads = []
for t_id in range(3):
    t = threading.Thread(target=log_worker, args=(t_id,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# -------------------------------
# Graceful shutdown
# -------------------------------
QEFFLogger.close_logger()
