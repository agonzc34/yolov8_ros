// MIT License
//
// Copyright (c) 2025 Alejandro González Cantón
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "yolo_cpp_ros/yolo_node.hpp"
#include "rclcpp/qos.hpp"
#include "yolo_cpp_ros/yolo/detect.hpp"
#include "yolo_cpp_ros/yolo/segment.hpp"
#include <string>

using namespace yolo_rclcpp;

YoloNode::YoloNode() : rclcpp_lifecycle::LifecycleNode("yolo_node") {}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_configure(const rclcpp_lifecycle::State &) {
  if (!this->params_declared) {
    this->declare_params();
    this->params_declared = true;
  }
  this->yolo_params = this->get_params();
  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_activate(const rclcpp_lifecycle::State &) {
  int image_reliability = this->yolo_params.image_reliability;
  rclcpp::ReliabilityPolicy qos_reliability_policy;
  if (image_reliability == 0) {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::SystemDefault;
  } else if (image_reliability == 1) {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::Reliable;
  } else {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::BestEffort;
  }
  auto img_sub_qos = rclcpp::QoS(1).reliability(qos_reliability_policy);

  this->detection_publisher =
      this->create_publisher<yolo_msgs::msg::DetectionArray>("detections",
                                                             rclcpp::QoS(10));
  this->image_subscription = this->create_subscription<sensor_msgs::msg::Image>(
      this->yolo_params.image_topic, img_sub_qos,
      std::bind(&YoloNode::recieve_image_callback, this,
                std::placeholders::_1));

  this->create_yolo(this->yolo_params);
  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_deactivate(const rclcpp_lifecycle::State &) {
  this->destroy_yolo();
  this->detection_publisher.reset();
  this->image_subscription.reset();
  RCLCPP_INFO(get_logger(), "[%s] Deactivated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_cleanup(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Cleaned up", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_shutdown(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Shutting down", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

void yolo_rclcpp::YoloNode::declare_params() {
  this->declare_parameter<std::string>("model", "yolo11m_segment.onnx");
  this->declare_parameter<std::string>("device", "cuda:0");
  this->declare_parameter<float>("threshold", 0.7);
  this->declare_parameter<float>("iou", 0.45);
  this->declare_parameter<int>("image_reliability", 2);
  this->declare_parameter<std::string>("image_topic", "image");
  this->declare_parameter<int>("n_threads", -1);
}

yolo_onnx_utils::YoloParams yolo_rclcpp::YoloNode::get_params() {
  yolo_onnx_utils::YoloParams params;
  this->get_parameter("model", params.model_path);
  this->get_parameter("device", params.device);
  this->get_parameter("threshold", params.threshold);
  this->get_parameter("iou", params.iou);
  this->get_parameter("image_reliability", params.image_reliability);
  this->get_parameter("image_topic", params.image_topic);
  this->get_parameter("n_threads", params.n_threads);
  return params;
}

void yolo_rclcpp::YoloNode::create_yolo(yolo_onnx_utils::YoloParams params) {
  if (params.model_path.find("segment") != std::string::npos) {
    this->yolo_model = std::make_unique<yolo_onnx::YoloSegment>(params);
  } else {
    this->yolo_model = std::make_unique<yolo_onnx::YoloDetect>(params);
  }
  RCLCPP_INFO(get_logger(), "[%s] Yolo model loaded", this->get_name());
}

void YoloNode::destroy_yolo() { this->yolo_model.reset(); }

void YoloNode::recieve_image_callback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  auto detection_array = yolo_msgs::msg::DetectionArray();

  if (this->yolo_model) {
    auto image = cv_bridge::toCvShare(msg, "bgr8")->image;
    auto detections = this->yolo_model->detect(image);
    detection_array.header = msg->header;
    detection_array.detections = detections;

    std::map<std::string, int> detections_per_class;
    for (const auto &detection : detections) {
      detections_per_class[detection.class_name]++;
    }

    if (detections.empty()) {
      RCLCPP_INFO(get_logger(), "No detections");
    } else {
      RCLCPP_INFO(get_logger(), "Total detections: %zu;%s", detections.size(),
                  std::accumulate(detections_per_class.begin(),
                                  detections_per_class.end(), std::string(),
                                  [](const std::string &a,
                                     const std::pair<std::string, int> &b) {
                                    return a + (a.empty() ? "" : ", ") + " - " +
                                           b.first + ": " +
                                           std::to_string(b.second);
                                  })
                      .c_str()); // TODO: sometimes it breaks here
    }

    // Publish detection array
    this->detection_publisher->publish(detection_array);
  }
}
