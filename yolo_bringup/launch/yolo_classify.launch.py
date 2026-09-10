# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    params_file = LaunchConfiguration("params_file")
    params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(
            get_package_share_directory("yolo_bringup"),
            "config",
            "yolo_classify.yaml",
        ),
        description="Path to the ROS 2 parameters file (YAML) with the "
        "config for the yolo_node block. All tuning (model, HF repo, "
        "topics, thresholds, QoS, top_k) lives here; the launch makes no "
        "topic remaps.",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo",
        description="Namespace for the nodes",
    )

    # C++ inference node (ONNX Runtime, GPU) running a classification model.
    # `model_type: Classify` in the params file forces the classification
    # postprocessor (image-level top-k classes on `classification`); the
    # model downloads from the Hugging Face Hub when model_repo +
    # model_filename are set. Classification has no boxes, so this launch
    # starts no tracking/3D/debug nodes.
    yolo_node_cmd = Node(
        package="yolo_ros",
        executable="yolo_node",
        name="yolo_node",
        namespace=namespace,
        parameters=[params_file],
    )

    return LaunchDescription(
        [
            params_file_cmd,
            namespace_cmd,
            yolo_node_cmd,
        ]
    )
