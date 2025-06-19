# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import fcntl
import math
import os
import re
import subprocess
import time
from typing import Optional

from QEfficient.utils.constants import LOCK_DIR, Constants
from QEfficient.utils.logging_utils import logger


def is_device_loaded(stdout: str) -> bool:
    try:
        match = re.search(r"Networks Loaded:(\d+)", stdout)
        return int(match.group(1)) > 0 if match else False

    except (ValueError, AttributeError):
        return False


def release_device_lock(lock_file):
    try:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    except Exception as e:
        logger.error(f"Error releasing lock: {e}")


def get_device_count():
    command = ["/opt/qti-aic/tools/qaic-util", "-q"]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        qids = re.findall(r"QID (\d+)", result.stdout)
        return max(map(int, qids)) + 1 if qids else 0

    except OSError:
        logger.warning("ERROR while fetching the device", command)
        return 0


def ensure_lock_dir(lock_dir: str):
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir)


def acquire_device_lock(retry_interval: int = 10, retry_duration: int = 300) -> Optional[object]:
    """
    Attempt to acquire a non-blocking exclusive lock on a device lock file.
    Retries every 10 seconds for up to 5 minutes.

    Args:
        device_id (int): The device ID to lock.

    Returns:
        file object if lock is acquired, else None.
    """
    ensure_lock_dir(LOCK_DIR)
    lock_file_path = os.path.join(LOCK_DIR, "device_check.lock")
    start_time = time.time()

    while (time.time() - start_time) < retry_duration:
        lock_file = open(lock_file_path, "w")

        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.debug("Lock acquired for device check")
            return lock_file

        except BlockingIOError:
            lock_file.close()
            logger.debug(f"Device check is locked. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

        except Exception as e:
            logger.error(f"Unexpected error acquiring lock for device check: {e}")
            return None

    logger.warning("Failed to acquire lock for device check after 5 minutes.")
    return None


def __fetch_device_id(device_count):
    for device_id in range(device_count):
        try:
            device_query_cmd = ["/opt/qti-aic/tools/qaic-util", "-q", "-d", str(device_id)]
            result = subprocess.run(device_query_cmd, capture_output=True, text=True)

            if "Failed to find requested device ID" in result.stdout:
                logger.warning(f"Device ID {device_id} not found.")
                continue

            if "Status:Error" in result.stdout or not is_device_loaded(result.stdout):
                logger.debug(f"Device {device_id} is not available.")
                continue

            logger.info(f"Device ID {device_id} is available and locked.")
            return [device_id]

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while querying device {device_id}.")
        except OSError as e:
            logger.error(f"OSError while querying device {device_id}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error while checking device {device_id}: {e}")
    return None


def get_available_device_id(retry_duration: int = 300, wait_time: int = 5) -> Optional[list[int]]:
    """
    Find an available Cloud AI 100 device ID using file-based locking.

    Args:
        max_retry_count (int): Maximum number of retries.
        wait_time (int): Seconds to wait between retries.

    Returns:
        list[int] | None: List containing available device ID, or None if not found.
    """
    device_count = get_device_count()

    if device_count == 0:
        logger.warning("No Cloud AI 100 devices found or platform SDK not installed.")
        return None

    lock_file = acquire_device_lock()

    if lock_file:
        start_time = time.time()

        while (time.time() - start_time) < retry_duration:
            device_id = __fetch_device_id(device_count)

            if device_id:
                release_device_lock(lock_file)
                return device_id

            time.sleep(wait_time)

    if lock_file:
        release_device_lock(lock_file)

    logger.warning("No available device found after all retries.")
    return None


def is_qpc_size_gt_32gb(params: int, mxfp6: bool) -> bool:
    if mxfp6:
        qpc_size = math.ceil((params * 1) / Constants.GB)
    else:
        qpc_size = math.ceil((params * 2) / Constants.GB)

    logger.warning(f"Approximate QPC size is: {qpc_size} GB")
    num_devices = math.ceil(qpc_size / Constants.MAX_QPC_LIMIT)
    logger.warning(f"Number of Devices required: {num_devices}")
    return qpc_size > Constants.MAX_QPC_LIMIT


def is_multi_qranium_setup_available():
    result = None
    command = ["/opt/qti-aic/tools/qaic-util", "-q"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, universal_newlines=True)
        filtered_result = subprocess.run(
            ["grep", "Device Capabilities"], input=result.stdout, stdout=subprocess.PIPE, text=True
        )
    except OSError:
        print("Command not found", command)
        return None

    lines = filtered_result.stdout.split("\n")

    # to count the number of devices in MQ enabled set up
    hybridboot_mdp_count = 0
    for line in lines:
        if ("HybridBoot+" in line) and ("MDP+" in line):
            hybridboot_mdp_count = hybridboot_mdp_count + 1

    if hybridboot_mdp_count > 0:
        print("No: of Devices with MQ enabled available: ", hybridboot_mdp_count)
        return True
    else:
        print("Device in MQ set up not available")
        return False
