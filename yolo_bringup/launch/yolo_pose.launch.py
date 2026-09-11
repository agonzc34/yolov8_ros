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
            "yolo_pose.yaml",
        ),
        description="Path to the ROS 2 parameters file (YAML) for the pose "
        "pipeline. Every parameter can also be overridden by the matching "
        "launch argument (run with --show-args for the list).",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo",
        description="Namespace for the nodes",
    )

    use_tracking = LaunchConfiguration("use_tracking")
    use_tracking_cmd = DeclareLaunchArgument(
        "use_tracking",
        default_value="True",
        description="Whether to enable the ByteTrack tracking node (True/False)",
    )

    use_3d = LaunchConfiguration("use_3d")
    use_3d_cmd = DeclareLaunchArgument(
        "use_3d",
        default_value="False",
        description="Whether to enable the 3D detection node (needs a depth "
        "image + CameraInfo, see the config file)",
    )

    use_debug = LaunchConfiguration("use_debug")
    use_debug_cmd = DeclareLaunchArgument(
        "use_debug",
        default_value="True",
        description="Whether to enable the debug/visualization node (True/False)",
    )

    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base),
        launch_arguments={
            "params_file": params_file,
            "namespace": namespace,
            "use_tracking": use_tracking,
            "use_3d": use_3d,
            "use_debug": use_debug,
        }.items(),
    )

    return LaunchDescription(
        [
            params_file_cmd,
            namespace_cmd,
            use_tracking_cmd,
            use_3d_cmd,
            use_debug_cmd,
            include,
        ]
    )
