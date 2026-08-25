# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

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
        description="Whether to enable the 3D detection node (needs a "
                    "depth image + CameraInfo, see the config file)",
    )

    params_file = LaunchConfiguration("params_file")
    params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(
            get_package_share_directory("yolo_bringup"),
            "config",
            "yolo_cpp.yaml",
        ),
        description="Path to the ROS 2 parameters file (YAML) with the config "
                    "for the yolo_node, tracking_node and debug_node blocks. "
                    "All tuning (model, topics, thresholds, QoS, tracker) lives "
                    "here; the launch makes no topic remaps.",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo",
        description="Namespace for the nodes",
    )

    # C++ inference node (ONNX Runtime, GPU). Everything — model, device,
    # image topic, thresholds, QoS, enable/max_det — comes from the params file.
    yolo_cpp_node_cmd = Node(
        package="yolo_cpp_ros",
        executable="yolo_cpp_ros",
        name="yolo_node",
        namespace=namespace,
        parameters=[params_file],
    )

    # C++ tracking node (ByteTrack, Kalman-filtered ids on `tracking`)
    tracking_node_cmd = Node(
        package="yolo_cpp_ros",
        executable="yolo_cpp_tracking",
        name="tracking_node",
        namespace=namespace,
        parameters=[params_file],
        condition=IfCondition(use_tracking),
    )

    # C++ 3D detection node (lifts the 2D detections to 3D using the depth
    # image; publishes on `detections_3d`). Topics and thresholds come from
    # the params file.
    detect_3d_node_cmd = Node(
        package="yolo_cpp_ros",
        executable="yolo_cpp_3d",
        name="detect_3d_node",
        namespace=namespace,
        parameters=[params_file],
        condition=IfCondition(use_3d),
    )

    # C++ debug node (visualizes detections/tracks + RViz 3D markers). Reads
    # the detections topic from the params file (`tracking` by default so the
    # tracked ids are shown when the tracking node is enabled).
    debug_node_cmd = Node(
        package="yolo_cpp_ros",
        executable="yolo_cpp_debug",
        name="debug_node",
        namespace=namespace,
        parameters=[params_file],
    )

    return LaunchDescription(
        [
            use_tracking_cmd,
            use_3d_cmd,
            params_file_cmd,
            namespace_cmd,
            yolo_cpp_node_cmd,
            tracking_node_cmd,
            detect_3d_node_cmd,
            debug_node_cmd,
        ]
    )
