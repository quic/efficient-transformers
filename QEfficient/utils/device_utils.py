# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
import re
import subprocess
from collections import defaultdict

from QEfficient.utils.constants import Constants
from QEfficient.utils.logging_utils import logger


def get_qaic_mdp_device_groups(min_nsp: int = 16, devices_per_group: int = 4) -> list[list[int]]:
    """Return ready, topology-compatible QAIC device groups suitable for parallel test workers."""
    command = ["/opt/qti-aic/tools/qaic-util", "-q"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        logger.warning("Not a Cloud AI 100 device, command not found: %s", command)
        return []

    if result.returncode != 0:
        logger.warning("Failed to query QAIC devices: %s", result.stderr.strip())
        return []

    groups = defaultdict(list)
    records = re.split(r"(?=^QID \d+\s*$)", result.stdout, flags=re.MULTILINE)
    for record in records:
        qid_match = re.search(r"^QID (\d+)\s*$", record, flags=re.MULTILINE)
        nsp_match = re.search(r"^\s*Nsp Total:(\d+)\s*$", record, flags=re.MULTILINE)
        board_match = re.search(r"^\s*Board serial:(.+?)\s*$", record, flags=re.MULTILINE)
        if not qid_match or not nsp_match or not board_match:
            continue
        if "Status:Ready" not in record or "HybridBoot+" not in record or "MDP+" not in record:
            continue
        if int(nsp_match.group(1)) < min_nsp:
            continue
        groups[board_match.group(1).strip()].append(int(qid_match.group(1)))

    device_groups = []
    for device_ids in groups.values():
        device_ids.sort()
        if len(device_ids) >= devices_per_group:
            device_groups.append(device_ids[:devices_per_group])
    return sorted(device_groups, key=lambda device_ids: device_ids[0])


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
