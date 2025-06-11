# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
import re
import subprocess
import time

from QEfficient.utils.constants import Constants
from QEfficient.utils.logging_utils import logger


def is_device_available(stdout: str) -> bool:
    try:
        match = re.search(r"Networks Loaded:(\d+)", stdout)
        return int(match.group(1)) > 0 if match else False
    except (ValueError, AttributeError):
        return False


def get_device_count():
    command = ["/opt/qti-aic/tools/qaic-util", "-q"]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        qids = re.findall(r"QID (\d+)", result.stdout)
        return max(map(int, qids)) + 1 if qids else 0
    except OSError:
        logger.warning("ERROR while fetching the device", command)
        return 0


def get_available_device_id(max_retry_count: int = 50, wait_time: int = 5) -> list[int] | None:
    """
    Find an available Cloud AI 100 device ID.

    Args:
    max_retry_count (int): Maximum number of retries.
    wait_time (int): Seconds to wait between retries.

    Returns:
    list[int] | None: List containing available device ID, or None if not found.
    """

    device_count = get_device_count()
    if device_count == 0:
        logger.warning("No Cloud AI 100 devices found or platform sdk not installed.")
        return None

    for retry_count in range(max_retry_count):
        for device_id in range(device_count):
            command = ["/opt/qti-aic/tools/qaic-util", "-q", "-d", str(device_id)]
            try:
                result = subprocess.run(command, capture_output=True, text=True)
            except OSError:
                logger.warning("Failed while querying the AIC card", command)
                return None

            if "Status:Error" in result.stdout or not is_device_available(result.stdout):
                continue

            elif "Status:Ready" in result.stdout:
                logger.info(f"Device ID : {device_id} is available.")
                return [device_id]

            elif "Failed to find requested device ID" in result.stdout:
                logger.warning("Device ID %d not found.", device_id)

        time.sleep(wait_time)
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
