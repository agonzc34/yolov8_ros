# Copyright (C) 2026 Alejandro González Cantón
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
            "yolo_cpp_pose.yaml",
        ),
        description="Path to the ROS 2 parameters file (YAML) with the config for "
                    "the yolo_node and debug_node blocks. All tuning (model, "
                    "topics, thresholds, QoS) lives here; the launch makes no "
                    "topic remaps.",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo_pose",
        description="Namespace for the nodes",
    )

    # C++ inference node (ONNX Runtime, GPU) running the pose model.
    # `model_type: Pose` in the params file forces the pose postprocessor
    # (bounding boxes + COCO keypoints) regardless of the model file name.
    yolo_cpp_node_cmd = Node(
        package="yolo_cpp_ros",
        executable="yolo_cpp_ros",
        name="yolo_node",
        namespace=namespace,
        parameters=[params_file],
    )

    # C++ debug node (visualizes detections + keypoint skeleton on the image)
    debug_node_cmd = Node(
        package="yolo_cpp_ros",
        executable="yolo_cpp_debug",
        name="debug_node",
        namespace=namespace,
        parameters=[params_file],
    )

    return LaunchDescription(
        [
            params_file_cmd,
            namespace_cmd,
            yolo_cpp_node_cmd,
            debug_node_cmd,
        ]
    )
