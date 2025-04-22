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

#ifndef YOLO_CPP_ROS__YOLO__DETECT_HPP_
#define YOLO_CPP_ROS__YOLO__DETECT_HPP_

#include "yolo_cpp_ros/engine/model.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include <vector>

namespace yolo_onnx {

std::vector<yolo_onnx_utils::Box> get_detection_without_nms(
    const std::vector<Ort::Value> &preds, std::vector<int64_t> output_shape,
    const cv::Size &original_image_size, const cv::Size &resized_image_size);

std::vector<yolo_onnx_utils::Box> get_detection_with_nms(
    const std::vector<Ort::Value> &preds, const cv::Size &original_image_size,
    const cv::Size &resized_image_size, const int num_classes,
    float iou_threshold, float conf_threshold);

class YoloDetect : public Model {
public:
  YoloDetect(yolo_onnx_utils::YoloParams params);
  ~YoloDetect();

protected:
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &outputTensors) override;
};
} // namespace yolo_onnx
#endif // YOLO_CPP_ROS__YOLO__DETECT_HPP_