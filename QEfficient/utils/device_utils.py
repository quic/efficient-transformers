# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
import subprocess

from QEfficient.utils.constants import Constants
from QEfficient.utils.logging_utils import logger


def get_available_device_id():
    """
    API to check available device id.

    Return:
        :int: Available device id.
    """

    device_id = 0
    result = None
    while 1:
        command = ["/opt/qti-aic/tools/qaic-util", "-q", "-d", f"{device_id}"]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except OSError:
            print("Not a Cloud AI 100 device, Command not found", command)
            return None
        if result:
            if "Status:Error" in result.stdout:
                device_id += 1
            elif "Status:Ready" in result.stdout:
                print("device is available.")
                return [device_id]
            elif "Failed to find requested device ID" in result.stdout:
                print("Failed to find requested device ID")
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
