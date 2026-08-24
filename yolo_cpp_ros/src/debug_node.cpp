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

#include "yolo_cpp_ros/debug_node.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/imgproc.hpp>
#include <string>

using namespace yolo_rclcpp;

DebugNode::DebugNode()
    : rclcpp_lifecycle::LifecycleNode("yolo_node"), class_to_color(),
      image_qos_profile(1) {
  this->declare_parameter("image_reliability", 2);
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Configuring...", this->get_name());

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

  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_activate(const rclcpp_lifecycle::State &) {
  this->image_subscription.subscribe(this->shared_from_this(), "image",
                                     image_qos_profile.get_rmw_qos_profile());
  this->detection_subscription.subscribe(
		this->shared_from_this(), "detections", image_qos_profile.get_rmw_qos_profile());

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

    auto return_image =
        cv_bridge::CvImage(msg_image->header, msg_image->encoding, image)
            .toImageMsg();
    this->debug_publisher->publish(*return_image.get());
  }
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

