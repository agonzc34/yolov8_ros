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

#include "yolo_cpp_ros/detect_3d_node.hpp"

#include <algorithm>
#include <cmath>

namespace yolo_rclcpp {

Detect3DNode::Detect3DNode()
    : rclcpp_lifecycle::LifecycleNode("detect_3d_node"),
      tf_buffer_(this->get_clock()) {
  this->declare_params();
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Configuring...", this->get_name());

  this->load_params();

  this->detections_3d_publisher_ =
      this->create_publisher<yolo_msgs::msg::DetectionArray>("detections_3d",
                                                             10);

  // The tf listener works off the executor's callbacks; it must be alive
  // from configuration onwards so the buffer stays populated.
  this->tf_listener_ = std::make_shared<tf2_ros::TransformListener>(
      this->tf_buffer_, this->shared_from_this());

  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::
      SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_activate(const rclcpp_lifecycle::State &) {
  auto reliability_to_policy = [](int r) {
    if (r == 0) {
      return rclcpp::ReliabilityPolicy::SystemDefault;
    }
    if (r == 1) {
      return rclcpp::ReliabilityPolicy::Reliable;
    }
    return rclcpp::ReliabilityPolicy::BestEffort;
  };

  rclcpp::QoS depth_image_qos = rclcpp::QoS(1).reliability(
      reliability_to_policy(this->depth_image_reliability_));
  rclcpp::QoS depth_info_qos = rclcpp::QoS(1).reliability(
      reliability_to_policy(this->depth_info_reliability_));

  this->depth_image_subscription_.subscribe(
      this->shared_from_this(), this->depth_image_topic_,
      depth_image_qos.get_rmw_qos_profile());
  this->depth_info_subscription_.subscribe(
      this->shared_from_this(), this->depth_info_topic_,
      depth_info_qos.get_rmw_qos_profile());
  this->detection_subscription_.subscribe(
      this->shared_from_this(), this->detections_topic_,
      rclcpp::QoS(10).get_rmw_qos_profile());

  this->synchronizer_ =
      std::make_shared<message_filters::Synchronizer<SyncPolicy3D>>(10);
  this->synchronizer_->connectInput(this->depth_image_subscription_,
                                    this->depth_info_subscription_,
                                    this->detection_subscription_);
  this->synchronizer_->registerCallback(
      std::bind(&Detect3DNode::recieve_callback, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3));

  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::
      SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_deactivate(const rclcpp_lifecycle::State &) {
  this->detection_subscription_.unsubscribe();
  this->depth_info_subscription_.unsubscribe();
  this->depth_image_subscription_.unsubscribe();
  this->synchronizer_.reset();

  RCLCPP_INFO(get_logger(), "[%s] Deactivated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::
      SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_cleanup(const rclcpp_lifecycle::State &) {
  this->tf_listener_.reset();
  this->detections_3d_publisher_.reset();

  RCLCPP_INFO(get_logger(), "[%s] Cleaned up", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::
      SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_shutdown(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Shutting down", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::
      SUCCESS;
}

void Detect3DNode::declare_params() {
  this->declare_parameter<std::string>("target_frame", "base_link");
  this->declare_parameter<double>("maximum_detection_threshold", 0.3);
  this->declare_parameter<int>("depth_image_units_divisor", 1000);
  this->declare_parameter<int>("depth_image_reliability", 2);
  this->declare_parameter<int>("depth_info_reliability", 2);
  this->declare_parameter<std::string>("depth_image_topic", "depth_image");
  this->declare_parameter<std::string>("depth_info_topic", "depth_info");
  this->declare_parameter<std::string>("detections_topic", "detections");
}

void Detect3DNode::load_params() {
  this->get_parameter("target_frame", this->target_frame_);
  this->get_parameter("maximum_detection_threshold",
                      this->maximum_detection_threshold_);
  this->get_parameter("depth_image_units_divisor",
                      this->depth_image_units_divisor_);
  this->get_parameter("depth_image_reliability",
                      this->depth_image_reliability_);
  this->get_parameter("depth_info_reliability", this->depth_info_reliability_);
  this->get_parameter("depth_image_topic", this->depth_image_topic_);
  this->get_parameter("depth_info_topic", this->depth_info_topic_);
  this->get_parameter("detections_topic", this->detections_topic_);
}

void Detect3DNode::recieve_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg) {
  yolo_msgs::msg::DetectionArray new_detections_msg;
  new_detections_msg.header = detections_msg->header;
  new_detections_msg.detections =
      this->process_detections(depth_msg, depth_info_msg, detections_msg);
  this->detections_3d_publisher_->publish(new_detections_msg);
}

std::vector<yolo_msgs::msg::Detection> Detect3DNode::process_detections(
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg) {
  std::vector<yolo_msgs::msg::Detection> new_detections;

  if (detections_msg->detections.empty()) {
    return new_detections;
  }

  auto transform =
      this->get_transform(depth_info_msg->header.frame_id);
  if (!transform) {
    return new_detections;
  }

  // Build an OpenCV wrapper over the depth image without converting it, like
  // the Python node's "passthrough" encoding: keep the raw 16UC1 millimeters
  // (or 32FC1 meters) and divide by depth_image_units_divisor below.
  cv::Mat depth_image;
  try {
    auto cv_ptr =
        cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    depth_image = cv_ptr->image;
  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return new_detections;
  }

  for (const auto &detection : detections_msg->detections) {
    auto bbox3d = this->convert_bb_to_3d(depth_image, *depth_info_msg,
                                         detection);
    if (!bbox3d) {
      continue;
    }

    yolo_msgs::msg::Detection new_detection = detection;
    new_detection.bbox3d =
        Detect3DNode::transform_3d_box(*bbox3d, transform->first,
                                       transform->second);
    new_detection.bbox3d.frame_id = this->target_frame_;
    new_detections.push_back(new_detection);

    if (!detection.keypoints.data.empty()) {
      auto keypoints3d = this->convert_keypoints_to_3d(
          depth_image, *depth_info_msg, detection);
      keypoints3d = Detect3DNode::transform_3d_keypoints(
          keypoints3d, transform->first, transform->second);
      keypoints3d.frame_id = this->target_frame_;
      new_detections.back().keypoints3d = keypoints3d;
    }
  }

  return new_detections;
}

std::optional<yolo_msgs::msg::BoundingBox3D> Detect3DNode::convert_bb_to_3d(
    const cv::Mat &depth_image, const sensor_msgs::msg::CameraInfo &depth_info,
    const yolo_msgs::msg::Detection &detection) {
  const int center_x = static_cast<int>(detection.bbox.center.position.x);
  const int center_y = static_cast<int>(detection.bbox.center.position.y);
  const int size_x = static_cast<int>(detection.bbox.size.x);
  const int size_y = static_cast<int>(detection.bbox.size.y);

  cv::Mat roi;
  if (!detection.mask.data.empty()) {
    // Crop the depth image by the segmentation mask polygon.
    cv::Mat mask = cv::Mat::zeros(depth_image.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours(1);
    contours[0].reserve(detection.mask.data.size());
    for (const auto &p : detection.mask.data) {
      contours[0].emplace_back(cvRound(p.x), cvRound(p.y));
    }
    cv::fillPoly(mask, contours, cv::Scalar(255));
    cv::bitwise_and(depth_image, depth_image, roi, mask);
  } else {
    // Crop the depth image by the 2D bounding box.
    const int u_min = std::max(center_x - size_x / 2, 0);
    const int u_max = std::min(center_x + size_x / 2, depth_image.cols - 1);
    const int v_min = std::max(center_y - size_y / 2, 0);
    const int v_max = std::min(center_y + size_y / 2, depth_image.rows - 1);
    if (u_max <= u_min || v_max <= v_min) {
      return std::nullopt;
    }
    roi = depth_image(cv::Rect(u_min, v_min, u_max - u_min, v_max - v_min));
  }

  // Convert to meters and bail out if there is no valid depth at all.
  cv::Mat roi_float;
  roi.convertTo(roi_float, CV_32FC1);
  roi_float /= static_cast<float>(this->depth_image_units_divisor_);
  if (cv::countNonZero(roi) == 0) {
    return std::nullopt;
  }

  double bb_center_z = 0.0;
  std::vector<float> roi_values;
  roi_values.reserve(roi_float.total());

  if (!detection.mask.data.empty()) {
    // Only the masked (nonzero) pixels take part in the z estimation.
    for (auto it = roi_float.begin<float>(); it != roi_float.end<float>();
         ++it) {
      if (*it > 0.0f) {
        roi_values.push_back(*it);
      }
    }
    if (roi_values.empty()) {
      return std::nullopt;
    }
    std::vector<float> sorted(roi_values);
    std::sort(sorted.begin(), sorted.end());
    const size_t mid = sorted.size() / 2;
    bb_center_z = (sorted.size() % 2 == 0)
                      ? 0.5 * (sorted[mid - 1] + sorted[mid])
                      : sorted[mid];
  } else {
    // Center pixel depth of the full image as the reference z.
    const int cy = std::clamp(center_y, 0, depth_image.rows - 1);
    const int cx = std::clamp(center_x, 0, depth_image.cols - 1);
    if (depth_image.type() == CV_16UC1) {
      bb_center_z = static_cast<double>(depth_image.at<uint16_t>(cy, cx)) /
                    this->depth_image_units_divisor_;
    } else {
      bb_center_z = static_cast<double>(depth_image.at<float>(cy, cx));
    }
    roi_values.insert(roi_values.end(), roi_float.begin<float>(),
                      roi_float.end<float>());
  }

  // Keep only the pixels close to the reference depth (same z-cluster).
  std::vector<float> filtered;
  filtered.reserve(roi_values.size());
  for (const float v : roi_values) {
    if (std::abs(v - bb_center_z) <= this->maximum_detection_threshold_) {
      filtered.push_back(v);
    }
  }
  if (filtered.empty()) {
    return std::nullopt;
  }

  const auto [z_min_it, z_max_it] =
      std::minmax_element(filtered.begin(), filtered.end());
  const double z = (*z_min_it + *z_max_it) / 2.0;
  if (z == 0.0) {
    return std::nullopt;
  }

  // Project from image space to camera frame using the intrinsics.
  const auto &k = depth_info.k;  // [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  const double fx = k[0], fy = k[4], px = k[2], py = k[5];

  yolo_msgs::msg::BoundingBox3D bbox3d;
  bbox3d.center.position.x = z * (center_x - px) / fx;
  bbox3d.center.position.y = z * (center_y - py) / fy;
  bbox3d.center.position.z = z;
  bbox3d.size.x = z * (size_x / fx);
  bbox3d.size.y = z * (size_y / fy);
  bbox3d.size.z = *z_max_it - *z_min_it;

  return bbox3d;
}

yolo_msgs::msg::KeyPoint3DArray Detect3DNode::convert_keypoints_to_3d(
    const cv::Mat &depth_image, const sensor_msgs::msg::CameraInfo &depth_info,
    const yolo_msgs::msg::Detection &detection) {
  const auto &k = depth_info.k;
  const double fx = k[0], fy = k[4], px = k[2], py = k[5];

  yolo_msgs::msg::KeyPoint3DArray keypoints3d;
  for (const auto &kp : detection.keypoints.data) {
    // Clamp the 2D keypoint into the image (the row index is the y pixel).
    const int u = std::clamp(static_cast<int>(kp.point.y), 0,
                             static_cast<int>(depth_info.height) - 1);
    const int v = std::clamp(static_cast<int>(kp.point.x), 0,
                             static_cast<int>(depth_info.width) - 1);

    double depth;
    if (depth_image.type() == CV_16UC1) {
      depth = static_cast<double>(depth_image.at<uint16_t>(u, v)) /
              this->depth_image_units_divisor_;  // 16UC1 raw units -> meters
    } else {
      depth = static_cast<double>(depth_image.at<float>(u, v));  // already m
    }
    if (!std::isfinite(depth)) {
      continue;
    }

    const double x = depth * (v - px) / fx;
    const double y = depth * (u - py) / fy;

    yolo_msgs::msg::KeyPoint3D kp3d;
    kp3d.id = kp.id;
    kp3d.score = kp.score;
    kp3d.point.x = x;
    kp3d.point.y = y;
    kp3d.point.z = depth;
    keypoints3d.data.push_back(kp3d);
  }

  return keypoints3d;
}

std::optional<std::pair<std::array<double, 3>, std::array<double, 4>>>
Detect3DNode::get_transform(const std::string &frame_id) {
  try {
    // Zero time = latest available transform (same as the Python node).
    const auto transform =
        this->tf_buffer_.lookupTransform(this->target_frame_, frame_id,
                                         tf2::TimePointZero);

    std::array<double, 3> translation{
        transform.transform.translation.x, transform.transform.translation.y,
        transform.transform.translation.z};
    std::array<double, 4> rotation{
        transform.transform.rotation.w, transform.transform.rotation.x,
        transform.transform.rotation.y, transform.transform.rotation.z};

    return std::make_pair(translation, rotation);
  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(get_logger(), "Could not transform: %s", ex.what());
    return std::nullopt;
  }
}

yolo_msgs::msg::BoundingBox3D Detect3DNode::transform_3d_box(
    const yolo_msgs::msg::BoundingBox3D &bbox,
    const std::array<double, 3> &translation,
    const std::array<double, 4> &rotation) {
  yolo_msgs::msg::BoundingBox3D out = bbox;

  // Position: rotate + translate.
  const auto position = Detect3DNode::qv_mult(
      rotation, {bbox.center.position.x, bbox.center.position.y,
                 bbox.center.position.z});
  out.center.position.x = position[0] + translation[0];
  out.center.position.y = position[1] + translation[1];
  out.center.position.z = position[2] + translation[2];

  // Size: only rotate (axis-aligned extents after the rotation).
  const auto size = Detect3DNode::qv_mult(
      rotation, {bbox.size.x, bbox.size.y, bbox.size.z});
  out.size.x = std::abs(size[0]);
  out.size.y = std::abs(size[1]);
  out.size.z = std::abs(size[2]);

  return out;
}

yolo_msgs::msg::KeyPoint3DArray Detect3DNode::transform_3d_keypoints(
    const yolo_msgs::msg::KeyPoint3DArray &keypoints,
    const std::array<double, 3> &translation,
    const std::array<double, 4> &rotation) {
  yolo_msgs::msg::KeyPoint3DArray out = keypoints;

  for (auto &point : out.data) {
    const auto position = Detect3DNode::qv_mult(
        rotation, {point.point.x, point.point.y, point.point.z});
    point.point.x = position[0] + translation[0];
    point.point.y = position[1] + translation[1];
    point.point.z = position[2] + translation[2];
  }

  return out;
}

std::array<double, 3> Detect3DNode::qv_mult(const std::array<double, 4> &q,
                                            const std::array<double, 3> &v) {
  const double qx = q[1], qy = q[2], qz = q[3], qw = q[0];

  // qvec = (qx, qy, qz); uv = qvec x v; uuv = qvec x uv
  const std::array<double, 3> uv{qy * v[2] - qz * v[1],
                                 qz * v[0] - qx * v[2],
                                 qx * v[1] - qy * v[0]};
  const std::array<double, 3> uuv{qy * uv[2] - qz * uv[1],
                                  qz * uv[0] - qx * uv[2],
                                  qx * uv[1] - qy * uv[0]};

  return {v[0] + 2.0 * (uv[0] * qw + uuv[0]),
          v[1] + 2.0 * (uv[1] * qw + uuv[1]),
          v[2] + 2.0 * (uv[2] * qw + uuv[2])};
}

}  // namespace yolo_rclcpp
