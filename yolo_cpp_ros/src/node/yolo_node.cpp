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

#include "yolo_cpp_ros/node/yolo_node.hpp"
#include "rclcpp/qos.hpp"
#include "yolo_cpp_ros/yolo/detect.hpp"
#include "yolo_cpp_ros/yolo/segment.hpp"
#include <algorithm>
#include <cctype>
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
  // Inference knobs mirroring the README / the Python yolo_node.py. A few are
  // accepted for config parity only (see get_params()): the C++/ONNX pipeline
  // always runs FP32 on the model's fixed input tensor and has no TTA.
  this->declare_parameter<std::string>("model_type", "auto");
  this->declare_parameter<std::string>("model", "yolo11m_segment.onnx");
  this->declare_parameter<std::string>("device", "cuda:0");
  this->declare_parameter<float>("threshold", 0.7);
  this->declare_parameter<float>("iou", 0.45);
  this->declare_parameter<int>("imgsz_height", 480);
  this->declare_parameter<int>("imgsz_width", 640);
  this->declare_parameter<bool>("half", false);
  this->declare_parameter<int>("max_det", 300);
  this->declare_parameter<bool>("augment", false);
  this->declare_parameter<bool>("agnostic_nms", false);
  this->declare_parameter<bool>("retina_masks", false);
  this->declare_parameter<bool>("enable", true);
  this->declare_parameter<int>("image_reliability", 2);
  this->declare_parameter<std::string>("image_topic", "image");
  this->declare_parameter<int>("n_threads", -1);
}

yolo_onnx_utils::YoloParams yolo_rclcpp::YoloNode::get_params() {
  yolo_onnx_utils::YoloParams params;
  this->get_parameter("model_type", params.model_type);
  this->get_parameter("model", params.model_path);
  this->get_parameter("device", params.device);
  this->get_parameter("threshold", params.threshold);
  this->get_parameter("iou", params.iou);
  this->get_parameter("enable", params.enable);
  this->get_parameter("max_det", params.max_det);
  this->get_parameter("image_reliability", params.image_reliability);
  this->get_parameter("image_topic", params.image_topic);
  this->get_parameter("n_threads", params.n_threads);
  return params;
}

void yolo_rclcpp::YoloNode::create_yolo(yolo_onnx_utils::YoloParams params) {
  std::string model_type = params.model_type;
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // An explicit model_type wins; "auto" (or empty) falls back to the
  // filename heuristic (path containing "segment" -> segmentation).
  const bool explicit_segment =
      !model_type.empty() && model_type != "auto" &&
      model_type.find("segment") != std::string::npos;
  const bool explicit_detect = model_type == "yolo" ||
                               model_type == "detect" ||
                               model_type == "det" || model_type == "detection";
  const bool by_filename =
      params.model_path.find("segment") != std::string::npos;

  if (explicit_segment || (by_filename && !explicit_detect)) {
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

  if (this->yolo_model && this->yolo_params.enable) {
    auto image = cv_bridge::toCvShare(msg, "bgr8")->image;
    auto detections = this->yolo_model->detect(image);

    // Cap published detections to max_det (post-NMS output is already
    // sorted by confidence, so this keeps the strongest max_det).
    if (static_cast<int>(detections.size()) > this->yolo_params.max_det) {
      detections.resize(this->yolo_params.max_det);
    }

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
