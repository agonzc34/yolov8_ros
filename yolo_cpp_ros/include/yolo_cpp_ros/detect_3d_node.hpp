// Copyright (C) 2026 Alejandro González Cantón
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef YOLO_CPP_ROS__DETECT_3D_NODE_HPP_
#define YOLO_CPP_ROS__DETECT_3D_NODE_HPP_

#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "yolo_msgs/msg/bounding_box3_d.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_msgs/msg/key_point3_d.hpp"
#include "yolo_msgs/msg/key_point3_d_array.hpp"

namespace yolo_rclcpp {

using SyncPolicy3D = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo,
    yolo_msgs::msg::DetectionArray>;

// C++-ONNX equivalent of yolo_ros/yolo_ros/detect_3d_node.py. Synchronizes
// the depth image, its CameraInfo and the 2D DetectionArray, lifts each
// 2D bbox (optionally using the segmentation mask to sample depth) into a
// BoundingBox3D, projects the pose keypoints into 3D, and transforms both
// into the target frame with tf2. Results are published on `detections_3d`.
class Detect3DNode : public rclcpp_lifecycle::LifecycleNode {
public:
  Detect3DNode();

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &);
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &);
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &);
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &);
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State &);

private:
  message_filters::Subscriber<sensor_msgs::msg::Image,
                              rclcpp_lifecycle::LifecycleNode>
      depth_image_subscription_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo,
                              rclcpp_lifecycle::LifecycleNode>
      depth_info_subscription_;
  message_filters::Subscriber<yolo_msgs::msg::DetectionArray,
                              rclcpp_lifecycle::LifecycleNode>
      detection_subscription_;
  rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr
      detections_3d_publisher_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy3D>> synchronizer_;

  std::string target_frame_;
  double maximum_detection_threshold_;
  int depth_image_units_divisor_;
  int depth_image_reliability_;
  int depth_info_reliability_;
  std::string depth_image_topic_;
  std::string depth_info_topic_;
  std::string detections_topic_;

  tf2_ros::Buffer tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  void declare_params();
  void load_params();

  void recieve_callback(
      const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg,
      const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg);

  std::vector<yolo_msgs::msg::Detection>
  process_detections(const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
                     const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg,
                     const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg);

  std::optional<yolo_msgs::msg::BoundingBox3D>
  convert_bb_to_3d(const cv::Mat &depth_image,
                   const sensor_msgs::msg::CameraInfo &depth_info,
                   const yolo_msgs::msg::Detection &detection);

  yolo_msgs::msg::KeyPoint3DArray
  convert_keypoints_to_3d(const cv::Mat &depth_image,
                          const sensor_msgs::msg::CameraInfo &depth_info,
                          const yolo_msgs::msg::Detection &detection);

  std::optional<std::pair<std::array<double, 3>, std::array<double, 4>>>
  get_transform(const std::string &frame_id);

  static yolo_msgs::msg::BoundingBox3D
  transform_3d_box(const yolo_msgs::msg::BoundingBox3D &bbox,
                   const std::array<double, 3> &translation,
                   const std::array<double, 4> &rotation);

  static yolo_msgs::msg::KeyPoint3DArray
  transform_3d_keypoints(const yolo_msgs::msg::KeyPoint3DArray &keypoints,
                         const std::array<double, 3> &translation,
                         const std::array<double, 4> &rotation);

  static std::array<double, 3> qv_mult(const std::array<double, 4> &q,
                                       const std::array<double, 3> &v);
};

}  // namespace yolo_rclcpp

#endif  // YOLO_CPP_ROS__DETECT_3D_NODE_HPP_
