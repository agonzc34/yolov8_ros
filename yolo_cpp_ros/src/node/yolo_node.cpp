// Copyright (c) 2025 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_cpp_ros/node/yolo_node.hpp"
#include "huggingface_hub.h"
#include "rclcpp/qos.hpp"
#include "yolo_cpp_ros/yolo/classify.hpp"
#include "yolo_cpp_ros/yolo/detect.hpp"
#include "yolo_cpp_ros/yolo/obb.hpp"
#include "yolo_cpp_ros/yolo/pose.hpp"
#include "yolo_cpp_ros/yolo/segment.hpp"
#include <algorithm>
#include <cctype>
#include <string>

namespace yolo_ros::node {

YoloNode::YoloNode() : rclcpp_lifecycle::LifecycleNode("yolo_node") {}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_configure(const rclcpp_lifecycle::State &) {
  if (!this->params_declared) {
    this->declare_params();
    this->params_declared = true;
  }
  this->yolo_params = this->get_params();
  this->enable_inference_.store(this->yolo_params.enable);
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

  // Runtime toggle for inference (matches the Python node's `enable` service).
  this->enable_service_ = this->create_service<std_srvs::srv::SetBool>(
      "enable", std::bind(&YoloNode::enable_service_callback, this,
                          std::placeholders::_1, std::placeholders::_2));

  this->create_yolo(this->yolo_params);
  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_deactivate(const rclcpp_lifecycle::State &) {
  this->destroy_yolo();
  this->enable_service_.reset();
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

void yolo_ros::node::YoloNode::declare_params() {
  // Inference knobs mirroring the README / the Python yolo_node.py (the
  // Python-only knobs imgsz_* / half / augment / agnostic_nms / retina_masks
  // were removed: the ONNX tensor fixes the input size, NMS is baked at
  // export, and the pipeline runs FP32 with no TTA).
  this->declare_parameter<std::string>("model_type", "auto");
  this->declare_parameter<std::string>("model", "yolo11m_segment.onnx");
  // Hugging Face Hub download: set model_repo + model_filename (and leave
  // `model` as anything) to fetch/reuse the model from the Hub at configure.
  this->declare_parameter<std::string>("model_repo", "");
  this->declare_parameter<std::string>("model_filename", "");
  this->declare_parameter<std::string>("cache_dir", "~/.cache/huggingface/hub");
  this->declare_parameter<bool>("force_download", false);
  this->declare_parameter<std::string>("device", "cuda:0");
  this->declare_parameter<float>("threshold", 0.7);
  this->declare_parameter<float>("iou", 0.45);
  this->declare_parameter<int>("max_det", 300);
  this->declare_parameter<bool>("enable", true);
  this->declare_parameter<int>("image_reliability", 2);
  this->declare_parameter<std::string>("image_topic", "image");
  this->declare_parameter<int>("n_threads", -1);
  this->declare_parameter<int>("max_fps", 0);
  this->declare_parameter<int>("top_k", 5);
}

yolo_ros::yolo::utils::YoloParams yolo_ros::node::YoloNode::get_params() {
  yolo_ros::yolo::utils::YoloParams params;
  this->get_parameter("model_type", params.model_type);
  this->get_parameter("model", params.model_path);
  this->get_parameter("model_repo", params.model_repo);
  this->get_parameter("model_filename", params.model_filename);
  this->get_parameter("cache_dir", params.cache_dir);
  this->get_parameter("force_download", params.force_download);
  this->get_parameter("device", params.device);
  this->get_parameter("threshold", params.threshold);
  this->get_parameter("iou", params.iou);
  this->get_parameter("enable", params.enable);
  this->get_parameter("max_det", params.max_det);
  this->get_parameter("image_reliability", params.image_reliability);
  this->get_parameter("image_topic", params.image_topic);
  this->get_parameter("n_threads", params.n_threads);
  this->get_parameter("max_fps", params.max_fps);
  this->get_parameter("top_k", params.top_k);

  // Hugging Face Hub: when an HF repo + filename are given, download (or reuse
  // the cached copy) and use that path instead of the local `model`.
  if (!params.model_repo.empty() && !params.model_filename.empty()) {
    auto result = huggingface_hub::hf_hub_download_with_shards(
        params.model_repo, params.model_filename, params.cache_dir,
        params.force_download);
    if (result.success) {
      params.model_path = result.path;
      RCLCPP_INFO(get_logger(), "[huggingface] model %s/%s -> %s",
                  params.model_repo.c_str(), params.model_filename.c_str(),
                  result.path.c_str());
    } else {
      RCLCPP_ERROR(get_logger(),
                   "[huggingface] failed to download %s/%s; falling back to "
                   "[local] %s",
                   params.model_repo.c_str(), params.model_filename.c_str(),
                   params.model_path.c_str());
    }
  } else {
    RCLCPP_INFO(get_logger(), "[local] model %s", params.model_path.c_str());
  }
  return params;
}

void yolo_ros::node::YoloNode::create_yolo(
    yolo_ros::yolo::utils::YoloParams params) {
  std::string model_type = params.model_type;
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // An explicit model_type wins; "auto" (or empty) falls back to the
  // filename heuristic (path containing "pose" -> pose, "segment" ->
  // segmentation, "cls"/"classify" -> classification, otherwise detection).
  const bool explicit_pose =
      !model_type.empty() && model_type != "auto" &&
      (model_type.find("pose") != std::string::npos ||
       model_type.find("keypoint") != std::string::npos || model_type == "kpt");
  const bool explicit_obb = !model_type.empty() && model_type != "auto" &&
                            (model_type.find("obb") != std::string::npos ||
                             model_type.find("rotat") != std::string::npos);
  const bool explicit_segment = !model_type.empty() && model_type != "auto" &&
                                model_type.find("segment") != std::string::npos;
  const bool explicit_detect = model_type == "yolo" || model_type == "detect" ||
                               model_type == "det" || model_type == "detection";
  const bool explicit_classify =
      !model_type.empty() && model_type != "auto" &&
      (model_type.find("class") != std::string::npos || model_type == "cls" ||
       model_type == "clf");
  const bool by_filename_pose =
      params.model_path.find("pose") != std::string::npos;
  const bool by_filename_obb =
      params.model_path.find("obb") != std::string::npos;
  const bool by_filename =
      params.model_path.find("segment") != std::string::npos;
  const bool by_filename_classify =
      params.model_path.find("cls") != std::string::npos ||
      params.model_path.find("classify") != std::string::npos;

  if (explicit_obb ||
      (by_filename_obb && !explicit_detect && !explicit_segment &&
       !explicit_pose && !explicit_classify && !by_filename_pose &&
       !by_filename && !by_filename_classify)) {
    this->yolo_model = std::make_unique<yolo_ros::yolo::YoloOBB>(params);
  } else if (explicit_pose ||
             (by_filename_pose && !explicit_detect && !explicit_segment &&
              !explicit_classify && !by_filename && !by_filename_classify)) {
    this->yolo_model = std::make_unique<yolo_ros::yolo::YoloPose>(params);
  } else if (explicit_segment ||
             (by_filename && !explicit_detect && !explicit_classify)) {
    this->yolo_model = std::make_unique<yolo_ros::yolo::YoloSegment>(params);
  } else if (explicit_classify || (by_filename_classify && !explicit_detect &&
                                   !explicit_segment && !by_filename)) {
    this->yolo_model = std::make_unique<yolo_ros::yolo::YoloClassify>(params);
  } else {
    this->yolo_model = std::make_unique<yolo_ros::yolo::YoloDetect>(params);
  }
  RCLCPP_INFO(get_logger(), "[%s] Yolo model loaded", this->get_name());
}

void YoloNode::destroy_yolo() { this->yolo_model.reset(); }

void YoloNode::enable_service_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
  this->enable_inference_.store(request->data);
  response->success = true;
  RCLCPP_INFO(get_logger(), "[%s] inference %s", this->get_name(),
              request->data ? "enabled" : "disabled");
}

void YoloNode::recieve_image_callback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  // Optional frequency cap (max_fps > 0): drop frames so inference and
  // publishing run at most max_fps Hz. The subscription stays live (DDS still
  // delivers every frame), we just skip the work for the frames in between.
  if (this->yolo_params.max_fps > 0) {
    const auto now = std::chrono::steady_clock::now();
    const double period_s = 1.0 / this->yolo_params.max_fps;
    if (std::chrono::duration<double>(now - this->last_inference_time_)
            .count() < period_s) {
      return; // too soon since the last processed frame: drop this one
    }
    this->last_inference_time_ = now;
  }

  auto detection_array = yolo_msgs::msg::DetectionArray();

  if (this->yolo_model && this->enable_inference_.load()) {
    // Convert to BGR8 (handles rgb8/mono8/... sources). toCvCopy throws on an
    // unsupported source encoding, so guard it to avoid crashing the callback.
    cv::Mat image;
    try {
      image =
          cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
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

} // namespace yolo_ros::node
