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


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():

    model = LaunchConfiguration("model")
    model_cmd = DeclareLaunchArgument(
        "model",
        default_value="/home/agonzc34/models/yolo11n-segment.onnx",
        description="Path to a segmentation ONNX model "
                    "(file name must contain 'segment' so the node selects YoloSegment)",
    )

    device = LaunchConfiguration("device")
    device_cmd = DeclareLaunchArgument(
        "device",
        default_value="cuda:0",
        description="Device to use (cuda:0 / cpu)",
    )

    image_topic = LaunchConfiguration("image_topic")
    image_topic_cmd = DeclareLaunchArgument(
        "image_topic",
        default_value="/webcam_image",
        description="Camera input topic",
    )

    image_reliability = LaunchConfiguration("image_reliability")
    image_reliability_cmd = DeclareLaunchArgument(
        "image_reliability",
        default_value="1",
        choices=["0", "1", "2"],
        description="QoS of the input image (0=system default, 1=Reliable, 2=Best Effort)",
    )

    threshold = LaunchConfiguration("threshold")
    threshold_cmd = DeclareLaunchArgument(
        "threshold",
        default_value="0.7",
        description="Minimum probability of a detection to be published",
    )

    iou = LaunchConfiguration("iou")
    iou_cmd = DeclareLaunchArgument(
        "iou",
        default_value="0.45",
        description="IoU threshold",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo_seg",
        description="Namespace for the nodes",
    )

    # Typed values for the C++ node (it declares strongly-typed params,
    # while launch args arrive as strings)
    threshold_value = PythonExpression(["float('", threshold, "')"])
    iou_value = PythonExpression(["float('", iou, "')"])
    image_reliability_value = PythonExpression(["int('", image_reliability, "')"])

    # C++ inference node (ONNX Runtime, GPU) running the segmentation model
    yolo_cpp_node_cmd = Node(
        package="yolo_cpp_ros",
        executable="yolo_cpp_ros",
        name="yolo_node",
        namespace=namespace,
        parameters=[
            {
                "model": model,
                "device": device,
                "threshold": threshold_value,
                "iou": iou_value,
                "image_reliability": image_reliability_value,
                "image_topic": image_topic,
            }
        ],
    )

    # C++ debug node (visualizes detections + masks on the image)
    debug_node_cmd = Node(
        package="yolo_cpp_ros",
        executable="yolo_cpp_debug",
        name="debug_node",
        namespace=namespace,
        parameters=[{"image_reliability": image_reliability_value}],
        remappings=[("image", image_topic)],
    )

    return LaunchDescription(
        [
            model_cmd,
            device_cmd,
            image_topic_cmd,
            image_reliability_cmd,
            threshold_cmd,
            iou_cmd,
            namespace_cmd,
            yolo_cpp_node_cmd,
            debug_node_cmd,
        ]
    )
