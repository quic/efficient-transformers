# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import cv2

video_path = "video_path"
output_size = (910, 512)  # Replace with your desired (x, y) dimensions

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

# Calculate frame indices to extract
num_frames_to_extract = 16
interval = int(total_frames / num_frames_to_extract)
frame_indices = [i * interval for i in range(num_frames_to_extract)]

# === Extract and resize frames ===
resized_frames = []

for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        resized = cv2.resize(frame, output_size)
        resized_frames.append(resized)
    else:
        print(f"Failed to read frame at index {idx}")

cap.release()

## Save frames ##
for i, frame in enumerate(resized_frames):
    cv2.imwrite(f"frame_{i + 1}.jpg", frame)

print(f"Extracted and resized {len(resized_frames)} frames to size {output_size}.")
