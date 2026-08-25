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

#include "yolo_cpp_ros/tracking_node.hpp"

#include <string>
#include <vector>

using namespace yolo_rclcpp;

TrackingNode::TrackingNode()
    : rclcpp_lifecycle::LifecycleNode("tracking_node"), image_qos_profile_(1) {
  this->declare_params();
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TrackingNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Configuring...", this->get_name());

  this->load_params();

  this->image_topic_ = this->get_parameter("image_topic").as_string();

  int image_reliability = this->get_parameter("image_reliability").as_int();
  rclcpp::ReliabilityPolicy qos_reliability_policy;
  if (image_reliability == 0) {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::SystemDefault;
  } else if (image_reliability == 1) {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::Reliable;
  } else {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::BestEffort;
  }
  this->image_qos_profile_ = rclcpp::QoS(1)
                                 .reliability(qos_reliability_policy)
                                 .durability_volatile()
                                 .keep_last(1);

  this->tracking_publisher_ =
      this->create_publisher<yolo_msgs::msg::DetectionArray>("tracking", 10);

  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TrackingNode::on_activate(const rclcpp_lifecycle::State &) {
  this->image_subscription_.subscribe(
      this->shared_from_this(), this->image_topic_,
      image_qos_profile_.get_rmw_qos_profile());
  this->detection_subscription_.subscribe(
      this->shared_from_this(), "detections",
      image_qos_profile_.get_rmw_qos_profile());

  this->synchronizer_ =
      std::make_shared<message_filters::Synchronizer<TrackingSyncPolicy>>(10);
  this->synchronizer_->connectInput(this->image_subscription_,
                                    this->detection_subscription_);
  this->synchronizer_->registerCallback(
      std::bind(&TrackingNode::recieve_callback, this, std::placeholders::_1,
                std::placeholders::_2));

  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TrackingNode::on_deactivate(const rclcpp_lifecycle::State &) {
  this->detection_subscription_.unsubscribe();
  this->image_subscription_.unsubscribe();
  this->synchronizer_.reset();
  RCLCPP_INFO(get_logger(), "[%s] Deactivated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TrackingNode::on_cleanup(const rclcpp_lifecycle::State &) {
  this->tracker_.reset();
  this->tracking_publisher_.reset();
  RCLCPP_INFO(get_logger(), "[%s] Cleaned up", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TrackingNode::on_shutdown(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Shutting down", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

void TrackingNode::declare_params() {
  // Same knobs as ultralytics bytetrack.yaml (default values).
  this->declare_parameter<int>("image_reliability", 2);
  this->declare_parameter<std::string>("image_topic", "image");
  this->declare_parameter<double>("track_high_thresh", 0.25);
  this->declare_parameter<double>("track_low_thresh", 0.1);
  this->declare_parameter<double>("new_track_thresh", 0.25);
  this->declare_parameter<int>("track_buffer", 30);
  this->declare_parameter<double>("match_thresh", 0.8);
  this->declare_parameter<bool>("fuse_score", true);
}

void TrackingNode::load_params() {
  this->get_parameter("track_high_thresh",
                      this->tracker_params_.track_high_thresh);
  this->get_parameter("track_low_thresh",
                      this->tracker_params_.track_low_thresh);
  this->get_parameter("new_track_thresh",
                      this->tracker_params_.new_track_thresh);
  this->get_parameter("track_buffer", this->tracker_params_.track_buffer);
  this->get_parameter("match_thresh", this->tracker_params_.match_thresh);
  this->get_parameter("fuse_score", this->tracker_params_.fuse_score);
  this->tracker_.reset();
  this->tracker_ = std::make_unique<yolo_tracking::ByteTrack>(
      this->tracker_params_);
}

void TrackingNode::recieve_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_image,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections) {
  yolo_msgs::msg::DetectionArray tracked_msg;
  tracked_msg.header = msg_image->header;

  // Convert the DetectionArray into the tracker's plain input format. The
  // detection index is preserved so the original Detection (class name, mask,
  // keypoints, ...) can be fetched back after tracking.
  std::vector<yolo_tracking::TrackDetection> dets;
  dets.reserve(msg_detections->detections.size());
  for (std::size_t i = 0; i < msg_detections->detections.size(); ++i) {
    const auto &det = msg_detections->detections[i];
    yolo_tracking::TrackDetection td;
    td.cx = det.bbox.center.position.x;
    td.cy = det.bbox.center.position.y;
    td.w = det.bbox.size.x;
    td.h = det.bbox.size.y;
    td.score = det.score;
    td.class_id = det.class_id;
    td.index = static_cast<int>(i);
    dets.push_back(td);
  }

  const auto tracks = this->tracker_->update(dets);

  for (const auto &track : tracks) {
    if (track.index < 0 ||
        track.index >= static_cast<int>(msg_detections->detections.size())) {
      continue;
    }
    // Copy the original detection (keeps class name, mask, etc.) and overlay
    // the Kalman-refined box and the stable track id.
    auto tracked_detection = msg_detections->detections[track.index];
    tracked_detection.bbox.center.position.x = (track.x1 + track.x2) / 2;
    tracked_detection.bbox.center.position.y = (track.y1 + track.y2) / 2;
    tracked_detection.bbox.size.x = track.x2 - track.x1;
    tracked_detection.bbox.size.y = track.y2 - track.y1;
    tracked_detection.id = std::to_string(track.id);

    tracked_msg.detections.push_back(tracked_detection);
  }

  this->tracking_publisher_->publish(tracked_msg);
}
