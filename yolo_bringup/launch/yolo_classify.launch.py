# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

from yolo_bringup.pipeline_launch import pipeline_launch

# Classification has no spatial output, so tracking and 3D stay off; the debug
# node is kept on to draw the image-level labels (empty bbox).
generate_launch_description = pipeline_launch(
    "classify",
    "Path to the ROS 2 parameters file (YAML) for the classification pipeline. "
    "Every yolo_node parameter can also be overridden by the matching launch "
    "argument (--show-args).",
    spatial=False,
)
