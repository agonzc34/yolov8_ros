// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__YOLO__SEGMENT_HPP_
#define YOLO_CPP_ROS__YOLO__SEGMENT_HPP_

#include "yolo_cpp_ros/engine/model.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include <vector>

namespace yolo_onnx {

std::vector<yolo_utils::BoxWithMask> get_segmentation_with_nms(
    const std::vector<Ort::Value> &preds, const cv::Size &original_image_size,
    const cv::Size &resized_image_size, const int num_classes,
    float iou_threshold, float conf_threshold);

class YoloSegment : public Model {
public:
  YoloSegment(yolo_utils::YoloParams params);
  ~YoloSegment();

protected:
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &outputTensors) override;
};
}  // namespace yolo_onnx
#endif // YOLO_CPP_ROS__YOLO__SEGMENT_HPP_