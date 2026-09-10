// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Instance segmentation task: detection boxes plus mask coefficients.

#ifndef YOLO_CPP_ROS__YOLO__SEGMENT_HPP_
#define YOLO_CPP_ROS__YOLO__SEGMENT_HPP_

#include "yolo_cpp_ros/engine/model.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include <vector>

/// @addtogroup yolo_tasks
/// @{
namespace yolo_onnx {

/// @brief Decode a raw segmentation tensor into boxes with mask coefficients
/// and apply the C++ per-class NMS.
/// @param[in] preds Raw output tensors from the model.
/// @param[in] original_image_size Size of the original camera image.
/// @param[in] resized_image_size Size of the letterboxed network input.
/// @param[in] num_classes Number of model classes.
/// @param[in] iou_threshold IoU threshold for NMS.
/// @param[in] conf_threshold Minimum score to keep a box.
/// @return NMS-filtered masked boxes in original-image coordinates.
std::vector<yolo_ros::yolo::utils::BoxWithMask> get_segmentation_with_nms(
    const std::vector<Ort::Value> &preds, const cv::Size &original_image_size,
    const cv::Size &resized_image_size, const int num_classes,
    float iou_threshold, float conf_threshold);

/// @brief YOLO instance segmentation model.
///
/// The exported graph emits a detection tensor plus prototype masks; the mask
/// coefficients are carried on each box and combined with the prototypes here.
class YoloSegment : public Model {
public:
  /// @brief Create the segmentation model from @p params.
  /// @param params Model and task configuration.
  YoloSegment(yolo_ros::yolo::utils::YoloParams params);
  /// @brief Destroy the segmentation model.
  ~YoloSegment();

protected:
  /// @brief Decode the segmentation tensors into Detection messages carrying
  /// a mask, applying NMS.
  /// @param[in] original_image_size Size of the original camera image.
  /// @param[in] resized_image_size Size of the letterboxed network input.
  /// @param[in] outputTensors Raw output tensors from inference().
  /// @return One Detection per kept instance, each with a Mask message.
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &outputTensors) override;
};
} // namespace yolo_onnx
/// @}
#endif // YOLO_CPP_ROS__YOLO__SEGMENT_HPP_
