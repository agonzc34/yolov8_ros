// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__NODE__TRACKING_NODE_HPP_
#define YOLO_CPP_ROS__NODE__TRACKING_NODE_HPP_

#include <memory>

#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "rclcpp/qos.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "yolo_cpp_ros/tracking/tracker.hpp"
#include "yolo_msgs/msg/detection_array.hpp"

namespace yolo_rclcpp {

using TrackingSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>;

// C++-ONNX equivalent of yolo_ros/yolo_ros/tracking_node.py. Consumes the
// synchronized image + DetectionArray pair, runs a BYTETracker over the
// detections and republishes the same detections on `tracking` with the
// Kalman-refined bounding boxes and stable track ids filled in.
class TrackingNode : public rclcpp_lifecycle::LifecycleNode {
public:
  TrackingNode();

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
      image_subscription_;
  message_filters::Subscriber<yolo_msgs::msg::DetectionArray,
                              rclcpp_lifecycle::LifecycleNode>
      detection_subscription_;
  rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr
      tracking_publisher_;
  std::shared_ptr<message_filters::Synchronizer<TrackingSyncPolicy>>
      synchronizer_;
  rclcpp::QoS image_qos_profile_;

  std::string image_topic_;

  std::unique_ptr<yolo_tracking::Tracker> tracker_;

  void declare_params();
  void load_params();

  void recieve_callback(
      const sensor_msgs::msg::Image::ConstSharedPtr &msg_image,
      const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections);
};

} // namespace yolo_rclcpp

#endif // YOLO_CPP_ROS__NODE__TRACKING_NODE_HPP_
