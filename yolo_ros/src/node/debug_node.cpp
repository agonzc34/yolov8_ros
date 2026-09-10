// Copyright (c) 2025 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/node/debug_node.hpp"
#include "cv_bridge/cv_bridge.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <opencv2/imgproc.hpp>
#include <string>

namespace yolo_ros::node {

DebugNode::DebugNode()
    : rclcpp_lifecycle::LifecycleNode("yolo_node"), image_qos_profile(1),
      class_to_color() {
  this->declare_parameter("image_reliability", 2);
  this->declare_parameter("image_topic", "image");
  this->declare_parameter("detections_topic", "detections");
  this->declare_parameter("markers_topic", "detections_3d");
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Configuring...", this->get_name());

  this->image_topic_ = this->get_parameter("image_topic").as_string();
  this->detections_topic_ = this->get_parameter("detections_topic").as_string();
  this->markers_topic_ = this->get_parameter("markers_topic").as_string();

  int image_reliability = this->get_parameter("image_reliability").as_int();
  rclcpp::ReliabilityPolicy qos_reliability_policy;
  if (image_reliability == 0) {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::SystemDefault;
  } else if (image_reliability == 1) {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::Reliable;
  } else {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::BestEffort;
  }
  this->image_qos_profile = rclcpp::QoS(1)
                                .reliability(qos_reliability_policy)
                                .durability_volatile()
                                .keep_last(1);

  this->debug_publisher =
      this->create_publisher<sensor_msgs::msg::Image>("debug_image", 10);

  this->bb_markers_publisher =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
          "debug_bb_markers", 10);

  this->kp_markers_publisher =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
          "debug_kp_markers", 10);

  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_activate(const rclcpp_lifecycle::State &) {
  this->image_subscription.subscribe(this->shared_from_this(),
                                     this->image_topic_,
                                     image_qos_profile.get_rmw_qos_profile());
  this->detection_subscription.subscribe(
      this->shared_from_this(), this->detections_topic_,
      image_qos_profile.get_rmw_qos_profile());

  uint32_t queue_size = 10;

  this->synchronizer =
      std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(
          queue_size);
  this->synchronizer->connectInput(this->image_subscription,
                                   this->detection_subscription);
  this->synchronizer->registerCallback(std::bind(&DebugNode::recieve_callback,
                                                 this, std::placeholders::_1,
                                                 std::placeholders::_2));

  // Separate subscription to the 3D-enriched stream purely for the RViz
  // markers. It is NOT part of the image sync, so a slow 3D/depth stream only
  // throttles the markers, never the debug image.
  this->markers_subscription_ =
      this->create_subscription<yolo_msgs::msg::DetectionArray>(
          this->markers_topic_, rclcpp::QoS(1),
          std::bind(&DebugNode::markers_callback, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_deactivate(const rclcpp_lifecycle::State &) {
  this->markers_subscription_.reset();
  RCLCPP_INFO(get_logger(), "[%s] Deactivated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_cleanup(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Cleaned up", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_shutdown(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Shutting down", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

void DebugNode::recieve_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_image,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections) {
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg_image, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;

  // Draw ALL detections onto one image, then publish ONCE (not per-detection).
  // Masks are drawn onto a single overlay layer and blended once, avoiding a
  // full image.clone() per detection.
  cv::Mat overlay = image.clone();
  for (const auto &detection : msg_detections->detections) {
    auto color = color_for_class(detection.class_name);
    image = draw_box(image, detection, color);
    draw_mask(overlay, image, detection, color);
    image = draw_keypoints(image, detection);
  }
  cv::addWeighted(overlay, 0.4, image, 0.6, 0, image);

  // The debug image is only gated by the image<->2D-detections sync, so it
  // publishes at the full 2D detection rate regardless of the 3D stream.
  // The Mat is always BGR8 (toCvCopy above forced BGR8 and OpenCV draws in
  // BGR), so advertise BGR8 — reusing the *original* encoding here mislabels
  // e.g. an rgb8 camera stream as rgb8 while the pixels are BGR, which makes
  // RViz swap red/blue and the image look blue-tainted.
  auto return_image =
      cv_bridge::CvImage(msg_image->header, sensor_msgs::image_encodings::BGR8,
                         image)
          .toImageMsg();
  this->debug_publisher->publish(*return_image.get());
}

// Drives the RViz 3D box / keypoint markers from the 3D-enriched stream. This
// runs on its OWN subscription (markers_topic_), so its publish rate is the 3D
// stream's rate (min(debug image, 3D detections)) and does not throttle the
// debug image.
void DebugNode::markers_callback(
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections) {
  visualization_msgs::msg::MarkerArray bb_marker_array;
  visualization_msgs::msg::MarkerArray kp_marker_array;

  for (const auto &detection : msg_detections->detections) {
    auto color = color_for_class(detection.class_name);

    // RViz markers for the 3D boxes (emitted only when a detect_3d node has
    // enriched the stream with bbox3d).
    if (!detection.bbox3d.frame_id.empty()) {
      auto marker = create_bb_marker(detection, color);
      marker.header.stamp = msg_detections->header.stamp;
      marker.id = bb_marker_array.markers.size();
      bb_marker_array.markers.push_back(marker);
    }

    // RViz markers for the 3D keypoints (pose output from a detect_3d node).
    if (!detection.keypoints3d.frame_id.empty()) {
      for (const auto &keypoint : detection.keypoints3d.data) {
        auto marker = create_kp_marker(keypoint);
        marker.header.frame_id = detection.keypoints3d.frame_id;
        marker.header.stamp = msg_detections->header.stamp;
        marker.id = kp_marker_array.markers.size();
        kp_marker_array.markers.push_back(marker);
      }
    }
  }

  this->bb_markers_publisher->publish(bb_marker_array);
  this->kp_markers_publisher->publish(kp_marker_array);
}

cv::Scalar DebugNode::color_for_class(const std::string &class_name) {
  auto color_it = class_to_color.find(class_name);
  if (color_it == class_to_color.end()) {
    // Deterministic FNV-1a hash of the class name so the same class always
    // gets the same color across runs (rand() made colors change every run).
    uint32_t hash = 2166136261u;
    for (const char c : class_name) {
      hash ^= static_cast<uint8_t>(c);
      hash *= 16777619u;
    }
    class_to_color[class_name] =
        cv::Scalar(hash & 0xFF, (hash >> 8) & 0xFF, (hash >> 16) & 0xFF);
    color_it = class_to_color.find(class_name);
  }
  return color_it->second;
}

cv::Mat DebugNode::draw_box(const cv::Mat &image,
                            const yolo_msgs::msg::Detection &detection,
                            const cv::Scalar &color) {
  const auto &center = detection.bbox.center.position;
  const double theta = detection.bbox.center.theta;

  // Text anchor, filled in below (top-left of the drawn box).
  int text_x = 0;
  int text_y = 0;

  if (std::abs(theta) > 0.0) {
    // Oriented bounding box (OBB): the rotation angle rides in center.theta
    // (radians, ultralytics xywhr convention). Draw the rotated quad with the
    // same corner geometry as the OBB postprocessor.
    const float c = static_cast<float>(std::cos(theta));
    const float s = static_cast<float>(std::sin(theta));
    const cv::Point2f ctr(static_cast<float>(center.x),
                          static_cast<float>(center.y));
    const float w = static_cast<float>(detection.bbox.size.x);
    const float h = static_cast<float>(detection.bbox.size.y);
    const cv::Point2f vec1(w / 2 * c, w / 2 * s);
    const cv::Point2f vec2(-h / 2 * s, h / 2 * c);
    const std::array<cv::Point2f, 4> corners = {
        ctr + vec1 + vec2, ctr + vec1 - vec2, ctr - vec1 - vec2,
        ctr - vec1 + vec2};

    float min_x = corners[0].x, min_y = corners[0].y;
    std::vector<cv::Point> quad;
    quad.reserve(4);
    for (const auto &p : corners) {
      quad.emplace_back(cvRound(p.x), cvRound(p.y));
      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
    }
    cv::polylines(image, quad, true, color, 2, cv::LINE_AA);
    text_x = cvRound(min_x);
    text_y = cvRound(min_y) - 5;
  } else {
    cv::Rect box(center.x - detection.bbox.size.x / 2,
                 center.y - detection.bbox.size.y / 2, detection.bbox.size.x,
                 detection.bbox.size.y);
    cv::rectangle(image, box, color, 2);
    text_x = box.x;
    text_y = box.y - 5;
  }

  // Text
  std::string text = detection.class_name;
  if (detection.id != "") {
    text += " " + detection.id;
  }
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(3) << detection.score;
  text += " " + ss.str();
  cv::putText(image, text, cv::Point(text_x, text_y), cv::FONT_HERSHEY_SIMPLEX,
              0.5, color, 2);

  return image;
}

void DebugNode::draw_mask(cv::Mat &overlay, cv::Mat &image,
                          const yolo_msgs::msg::Detection &detection,
                          const cv::Scalar &color) {
  if (detection.mask.data.size() == 0) {
    return;
  }
  // Convert ROS Point2D (float64) mask boundary points to OpenCV points
  std::vector<std::vector<cv::Point>> contours(1);
  contours[0].reserve(detection.mask.data.size());
  for (const auto &p : detection.mask.data) {
    contours[0].emplace_back(cvRound(p.x), cvRound(p.y));
  }
  // Fill the mask onto the shared overlay layer (blended once by the caller)
  // and draw the crisp outline directly on the final image.
  cv::fillPoly(overlay, contours, color, cv::LINE_AA);
  cv::polylines(image, contours, true, color, 2, cv::LINE_AA);
}

cv::Mat DebugNode::draw_keypoints(const cv::Mat &image,
                                  const yolo_msgs::msg::Detection &detection) {
  if (detection.keypoints.data.size() == 0) {
    return image;
  }

  // COCO human pose skeleton, using the 1-based keypoint ids published by
  // yolo_node.
  static const int skeleton[19][2] = {
      {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
      {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
      {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}};

  // Generate a stable BGR color directly from the keypoint or limb index.
  // This keeps adjacent parts visually distinct without a borrowed palette.
  auto indexed_color = [](int index) {
    const int i = ((index % 256) + 256) % 256;
    return cv::Scalar((53 * i + 67) % 256, (97 * i + 149) % 256,
                      (193 * i + 43) % 256);
  };

  std::map<int, cv::Point> points;
  for (const auto &kp : detection.keypoints.data) {
    const auto pt = cv::Point(cvRound(kp.point.x), cvRound(kp.point.y));
    points[kp.id] = pt;
    const auto color_k = indexed_color(kp.id);
    cv::circle(image, pt, 5, color_k, -1, cv::LINE_AA);
    cv::putText(image, std::to_string(kp.id), pt, cv::FONT_HERSHEY_SIMPLEX, 1,
                color_k, 1, cv::LINE_AA);
  }

  // Draw the skeleton limbs on top (per-limb colors, only when both
  // endpoints of the limb are present in the detection).
  for (int i = 0; i < 19; ++i) {
    auto it1 = points.find(skeleton[i][0]);
    auto it2 = points.find(skeleton[i][1]);
    if (it1 != points.end() && it2 != points.end()) {
      cv::line(image, it1->second, it2->second, indexed_color(i + 32), 2,
               cv::LINE_AA);
    }
  }
  return image;
}

visualization_msgs::msg::Marker
DebugNode::create_bb_marker(const yolo_msgs::msg::Detection &detection,
                            const cv::Scalar &color) {
  visualization_msgs::msg::Marker marker;

  marker.header.frame_id = detection.bbox3d.frame_id;
  marker.ns = "yolo_3d";
  marker.type = visualization_msgs::msg::Marker::CUBE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.frame_locked = false;

  marker.pose.position.x = detection.bbox3d.center.position.x;
  marker.pose.position.y = detection.bbox3d.center.position.y;
  marker.pose.position.z = detection.bbox3d.center.position.z;
  marker.pose.orientation.x = detection.bbox3d.center.orientation.x;
  marker.pose.orientation.y = detection.bbox3d.center.orientation.y;
  marker.pose.orientation.z = detection.bbox3d.center.orientation.z;
  marker.pose.orientation.w = detection.bbox3d.center.orientation.w;

  marker.scale.x = detection.bbox3d.size.x;
  marker.scale.y = detection.bbox3d.size.y;
  marker.scale.z = detection.bbox3d.size.z;

  // The per-class color is stored as an OpenCV BGR scalar -> convert to RGB.
  marker.color.r = color[2] / 255.0;
  marker.color.g = color[1] / 255.0;
  marker.color.b = color[0] / 255.0;
  marker.color.a = 0.4;

  marker.lifetime.sec = 0;
  marker.lifetime.nanosec = 0; // persistent: stays visible until next update
  marker.text = detection.class_name;

  return marker;
}

visualization_msgs::msg::Marker
DebugNode::create_kp_marker(const yolo_msgs::msg::KeyPoint3D &keypoint) {
  visualization_msgs::msg::Marker marker;

  marker.ns = "yolo_3d";
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.frame_locked = false;

  marker.pose.position.x = keypoint.point.x;
  marker.pose.position.y = keypoint.point.y;
  marker.pose.position.z = keypoint.point.z;
  marker.pose.orientation.w = 1.0;

  marker.scale.x = 0.05;
  marker.scale.y = 0.05;
  marker.scale.z = 0.05;

  // Confidence gradient from red (low score) to blue (high score).
  marker.color.r = 1.0 - keypoint.score;
  marker.color.g = 0.0;
  marker.color.b = keypoint.score;
  marker.color.a = 0.4;

  marker.lifetime.sec = 0;
  marker.lifetime.nanosec = 0; // persistent: stays visible until next update
  marker.text = std::to_string(keypoint.id);

  return marker;
}

} // namespace yolo_ros::node
