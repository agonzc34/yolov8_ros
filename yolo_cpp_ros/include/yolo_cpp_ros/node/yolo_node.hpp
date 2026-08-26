// Copyright (c) 2025 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__NODE__YOLO_NODE_HPP_
#define YOLO_CPP_ROS__NODE__YOLO_NODE_HPP_

#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include <atomic>
#include <chrono>
#include <memory>

#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "yolo_cpp_ros/engine/model.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"

namespace yolo_rclcpp {
class YoloNode : public rclcpp_lifecycle::LifecycleNode {
public:
  YoloNode();

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

protected:
  std::unique_ptr<yolo_onnx::Model> yolo_model;

  rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr
      detection_publisher;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription;

  void declare_params();
  yolo_onnx_utils::YoloParams get_params();

  yolo_onnx_utils::YoloParams yolo_params;
  bool params_declared = false;

  // Runtime inference gate, toggled by the `enable` service (SetBool); the
  // `enable` parameter only provides the initial value. Atomic because the
  // image subscription and the service may run on different executor threads.
  std::atomic<bool> enable_inference_{true};

  // Timestamp of the last processed frame, used by the max_fps frequency cap
  // to decide whether the current frame should be dropped.
  std::chrono::steady_clock::time_point last_inference_time_{};

  void create_yolo(yolo_onnx_utils::YoloParams params);
  void destroy_yolo();

private:
  void recieve_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void enable_service_callback(
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_service_;
};
} // namespace yolo_rclcpp

#endif // YOLO_CPP_ROS__NODE__YOLO_NODE_HPP_