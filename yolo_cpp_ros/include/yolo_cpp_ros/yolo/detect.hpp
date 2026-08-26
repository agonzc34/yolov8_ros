// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__YOLO__DETECT_HPP_
#define YOLO_CPP_ROS__YOLO__DETECT_HPP_

#include "yolo_cpp_ros/engine/model.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include <vector>

namespace yolo_onnx {

std::vector<yolo_utils::Box> get_detection_without_nms(
    const std::vector<Ort::Value> &preds, std::vector<int64_t> output_shape,
    const cv::Size &original_image_size, const cv::Size &resized_image_size);

std::vector<yolo_utils::Box> get_detection_with_nms(
    const std::vector<Ort::Value> &preds, const cv::Size &original_image_size,
    const cv::Size &resized_image_size, const int num_classes,
    float iou_threshold, float conf_threshold);

class YoloDetect : public Model {
public:
  YoloDetect(yolo_utils::YoloParams params);
  ~YoloDetect();

protected:
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &outputTensors) override;
};
}  // namespace yolo_onnx
#endif // YOLO_CPP_ROS__YOLO__DETECT_HPP_