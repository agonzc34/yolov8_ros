// Copyright (c) 2025 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__NODE__DEBUG_NODE_HPP_
#define YOLO_CPP_ROS__NODE__DEBUG_NODE_HPP_

#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include "message_filters/subscriber.h"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_msgs/msg/key_point3_d.hpp"

#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

#include <opencv2/opencv.hpp>
#include <rclcpp/context.hpp>

namespace yolo_rclcpp {
using ApproximateSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>;

class DebugNode : public rclcpp_lifecycle::LifecycleNode {
public:
  DebugNode();

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
      image_subscription;
  message_filters::Subscriber<yolo_msgs::msg::DetectionArray,
                              rclcpp_lifecycle::LifecycleNode>
      detection_subscription;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_publisher;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      bb_markers_publisher;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      kp_markers_publisher;
  std::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>>
      synchronizer;

  rclcpp::QoS image_qos_profile;

  std::string image_topic_;
  std::string detections_topic_;
  std::string markers_topic_;

  // Independently subscribes to the 3D-enriched stream (detections_3d) purely
  // to drive the RViz 3D markers. Keeping this separate from the
  // image<->2D-detections sync means debug_image publishes at the full 2D
  // detection rate while the (slower) 3D stream only gates the markers.
  rclcpp::Subscription<yolo_msgs::msg::DetectionArray>::SharedPtr
      markers_subscription_;

  std::map<std::string, cv::Scalar> class_to_color;

  void recieve_callback(
      const sensor_msgs::msg::Image::ConstSharedPtr &msg_image,
      const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections);
  void markers_callback(
      const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections);

  cv::Scalar color_for_class(const std::string &class_name);

  cv::Mat draw_box(const cv::Mat &image,
                   const yolo_msgs::msg::Detection &detection,
                   const cv::Scalar &color);
	cv::Mat draw_mask(const cv::Mat &image,
									 const yolo_msgs::msg::Detection &detection, const cv::Scalar &color);
	cv::Mat draw_keypoints(const cv::Mat &image,
												 const yolo_msgs::msg::Detection &detection);

	visualization_msgs::msg::Marker create_bb_marker(
			const yolo_msgs::msg::Detection &detection,
			const cv::Scalar &color);
	visualization_msgs::msg::Marker create_kp_marker(
			const yolo_msgs::msg::KeyPoint3D &keypoint);

					};
}  // namespace yolo_rclcpp

#endif // YOLO_CPP_ROS__NODE__DEBUG_NODE_HPP_