# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

from yolo_bringup.pipeline_launch import pipeline_launch

generate_launch_description = pipeline_launch(
    "obb",
    "Path to the ROS 2 parameters file (YAML) for the OBB pipeline. Every "
    "parameter can also be overridden by the matching launch argument (run "
    "with --show-args for the list).",
)
