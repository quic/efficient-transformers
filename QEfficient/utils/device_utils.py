# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import subprocess


def get_available_device_id():
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
