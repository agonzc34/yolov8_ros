// Copyright (C) 2025 Alejandro González Cantón
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

#ifndef YOLO_CPP_ROS__NODE__YOLO_NODE_HPP_
#define YOLO_CPP_ROS__NODE__YOLO_NODE_HPP_

#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include <memory>

#include "sensor_msgs/msg/image.hpp"
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

  void create_yolo(yolo_onnx_utils::YoloParams params);
  void destroy_yolo();

private:
  void recieve_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
};
} // namespace yolo_rclcpp

#endif // YOLO_CPP_ROS__NODE__YOLO_NODE_HPP_