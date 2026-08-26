// Copyright (c) 2025 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_cpp_ros/node/debug_node.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/imgproc.hpp>
#include <map>
#include <string>

using namespace yolo_rclcpp;

DebugNode::DebugNode()
    : rclcpp_lifecycle::LifecycleNode("yolo_node"), image_qos_profile(1),
      class_to_color() {
  this->declare_parameter("image_reliability", 2);
  this->declare_parameter("image_topic", "image");
  this->declare_parameter("detections_topic", "detections");
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Configuring...", this->get_name());

  this->image_topic_ = this->get_parameter("image_topic").as_string();
  this->detections_topic_ = this->get_parameter("detections_topic").as_string();

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
  this->image_subscription.subscribe(this->shared_from_this(), this->image_topic_,
                                     image_qos_profile.get_rmw_qos_profile());
  this->detection_subscription.subscribe(
		this->shared_from_this(), this->detections_topic_, image_qos_profile.get_rmw_qos_profile());

  uint32_t queue_size = 10;

  this->synchronizer =
      std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(
						queue_size);
  this->synchronizer->connectInput(this->image_subscription,
                                   this->detection_subscription);
  this->synchronizer->registerCallback(std::bind(&DebugNode::recieve_callback,
                                                 this, std::placeholders::_1,
                                                 std::placeholders::_2));

  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_deactivate(const rclcpp_lifecycle::State &) {
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
    cv_ptr = cv_bridge::toCvCopy(msg_image,
    sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;

  visualization_msgs::msg::MarkerArray bb_marker_array;
  visualization_msgs::msg::MarkerArray kp_marker_array;

  for (const auto &detection : msg_detections->detections) {
    auto class_name = detection.class_name;
    auto color_it = class_to_color.find(class_name);
    if (color_it == class_to_color.end()) {
      class_to_color[class_name] =
          cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
      color_it = class_to_color.find(class_name);
    }
    cv::Scalar color = color_it->second;

    image = draw_box(image, detection, color);
    image = draw_mask(image, detection, color);
    image = draw_keypoints(image, detection);

    // RViz markers for the 3D boxes (emitted only when a detect_3d node has
    // enriched the detection stream with bbox3d).
    if (!detection.bbox3d.frame_id.empty()) {
      auto marker = create_bb_marker(detection, color);
      marker.header.stamp = msg_image->header.stamp;
      marker.id = bb_marker_array.markers.size();
      bb_marker_array.markers.push_back(marker);
    }

    // RViz markers for the 3D keypoints (pose output from a detect_3d node).
    if (!detection.keypoints3d.frame_id.empty()) {
      for (const auto &keypoint : detection.keypoints3d.data) {
        auto marker = create_kp_marker(keypoint);
        marker.header.frame_id = detection.keypoints3d.frame_id;
        marker.header.stamp = msg_image->header.stamp;
        marker.id = kp_marker_array.markers.size();
        kp_marker_array.markers.push_back(marker);
      }
    }
  }

  // Publish ONCE with all detections/masks drawn (not once per detection).
  // The Mat is always BGR8 (toCvCopy above forced BGR8 and OpenCV draws in
  // BGR), so advertise BGR8 — reusing the *original* encoding here mislabels
  // e.g. an rgb8 camera stream as rgb8 while the pixels are BGR, which makes
  // RViz swap red/blue and the image look blue-tainted.
  auto return_image =
      cv_bridge::CvImage(msg_image->header, sensor_msgs::image_encodings::BGR8,
                         image).toImageMsg();
  this->debug_publisher->publish(*return_image.get());

  this->bb_markers_publisher->publish(bb_marker_array);
  this->kp_markers_publisher->publish(kp_marker_array);
}

cv::Mat DebugNode::draw_box(const cv::Mat &image,
														 const yolo_msgs::msg::Detection &detection,
														 const cv::Scalar &color) {
	cv::Rect box(detection.bbox.center.position.x - detection.bbox.size.x / 2,
								detection.bbox.center.position.y - detection.bbox.size.y / 2,
								detection.bbox.size.x, detection.bbox.size.y);
	cv::rectangle(image, box, color, 2);

	// Text
	std::string text = detection.class_name;
	if (detection.id != "") {
		text += " " + detection.id;
	}
	std::ostringstream ss;
	ss << std::fixed << std::setprecision(3) << detection.score;
	text += " " + ss.str();
	cv::putText(image, text, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX,
							0.5, color, 2);

	return image;
}

cv::Mat DebugNode::draw_mask(const cv::Mat &image,
	const yolo_msgs::msg::Detection &detection,
	const cv::Scalar &color) {
		if (detection.mask.data.size() == 0) {
			return image;
		}
		auto layer = image.clone();
		// Convert ROS Point2D (float64) mask boundary points to OpenCV points
		std::vector<std::vector<cv::Point>> contours(1);
		contours[0].reserve(detection.mask.data.size());
		for (const auto &p : detection.mask.data) {
			contours[0].emplace_back(cvRound(p.x), cvRound(p.y));
		}
		cv::fillPoly(layer, contours,
										color, cv::LINE_AA);
		cv::addWeighted(layer, 0.4, image, 0.6, 0, image);
		cv::polylines(image, contours,
										true, color, 2, cv::LINE_AA);
		return image;
	}

cv::Mat DebugNode::draw_keypoints(const cv::Mat &image,
																	const yolo_msgs::msg::Detection &detection) {
	if (detection.keypoints.data.size() == 0) {
		return image;
	}

	// COCO human pose skeleton, using the 1-based keypoint ids published by
	// yolo_node.
	static const int skeleton[19][2] = {{16, 14}, {14, 12}, {17, 15}, {15, 13},
	                                    {12, 13}, {6, 12},  {7, 13},  {6, 7},
	                                    {6, 8},   {7, 9},   {8, 10},  {9, 11},
	                                    {2, 3},   {1, 2},   {1, 3},   {2, 4},
	                                    {3, 5},   {4, 6},   {5, 7}};

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
		cv::putText(image, std::to_string(kp.id), pt, cv::FONT_HERSHEY_SIMPLEX,
		            1, color_k, 1, cv::LINE_AA);
	}

	// Draw the skeleton limbs on top (per-limb colors, only when both
	// endpoints of the limb are present in the detection).
	for (int i = 0; i < 19; ++i) {
		auto it1 = points.find(skeleton[i][0]);
		auto it2 = points.find(skeleton[i][1]);
		if (it1 != points.end() && it2 != points.end()) {
			cv::line(image, it1->second, it2->second, indexed_color(i + 32),
			         2, cv::LINE_AA);
		}
	}
	return image;
}

visualization_msgs::msg::Marker DebugNode::create_bb_marker(
    const yolo_msgs::msg::Detection &detection,
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
  marker.pose.orientation.w = 1.0;

  marker.scale.x = detection.bbox3d.size.x;
  marker.scale.y = detection.bbox3d.size.y;
  marker.scale.z = detection.bbox3d.size.z;

  // The per-class color is stored as an OpenCV BGR scalar -> convert to RGB.
  marker.color.r = color[2] / 255.0;
  marker.color.g = color[1] / 255.0;
  marker.color.b = color[0] / 255.0;
  marker.color.a = 0.4;

  marker.lifetime.sec = 0;
  marker.lifetime.nanosec = 500000000;
  marker.text = detection.class_name;

  return marker;
}

visualization_msgs::msg::Marker DebugNode::create_kp_marker(
    const yolo_msgs::msg::KeyPoint3D &keypoint) {
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
  marker.lifetime.nanosec = 500000000;
  marker.text = std::to_string(keypoint.id);

  return marker;
}

