# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    base = os.path.join(
        get_package_share_directory("yolo_bringup"), "launch", "yolo.launch.py"
    )

    params_file = LaunchConfiguration("params_file")
    params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(
            get_package_share_directory("yolo_bringup"),
            "config",
            "yolo_classify.yaml",
        ),
        description="Path to the ROS 2 parameters file (YAML) for the "
        "classification pipeline. Every yolo_node parameter can also be "
        "overridden by the matching launch argument (--show-args).",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo",
        description="Namespace for the nodes",
    )

    # Classification has no spatial output: start yolo_node only.
    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base),
        launch_arguments={
            "params_file": params_file,
            "namespace": namespace,
            "use_tracking": "False",
            "use_3d": "False",
            "use_debug": "False",
        }.items(),
    )

    return LaunchDescription([params_file_cmd, namespace_cmd, include])
