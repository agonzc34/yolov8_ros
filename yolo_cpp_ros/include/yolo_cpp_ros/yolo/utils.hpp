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

#ifndef YOLO_CPP_ROS__YOLO__UTILS_HPP_
#define YOLO_CPP_ROS__YOLO__UTILS_HPP_

#include "onnxruntime_cxx_api.h"
#include "yolo_msgs/msg/bounding_box2_d.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace yolo_onnx_utils {
struct Box {
  float x1, y1, x2, y2, score;
  int index, class_id;

  Box(float x1, float y1, float x2, float y2, float score, int index,
      int class_id)
      : x1(x1), y1(y1), x2(x2), y2(y2), score(score), index(index),
        class_id(class_id) {}
  Box() = default;
  virtual ~Box() = default;
};

struct BoxWithMask : public Box {
  std::vector<float> mask_coeffs;

  BoxWithMask() = default;
  BoxWithMask(Box box)
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index, box.class_id),
        mask_coeffs(std::vector<float>()) {}
  BoxWithMask(Box box, std::vector<float> mask_coeffs)
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index, box.class_id),
        mask_coeffs(mask_coeffs) {}
};

struct YoloParams {
  std::string model_path;
  std::string device;
  float threshold;
  float iou;
  int image_reliability;
  std::string image_topic;
  int n_threads;
};

float iou(const std::shared_ptr<Box> &box1, const std::shared_ptr<Box> &box2);

std::vector<int> nms(std::vector<std::shared_ptr<Box>> &boxes,
                     float iou_threshold, float conf_threshold);
cv::Mat letterbox(const cv::Mat &img, const cv::Size &new_shape,
                  const cv::Scalar &color);
cv::Mat inverse_letterbox(const cv::Mat &letterboxed,
                          const cv::Size &original_image_size,
                          const cv::Size &resized_image_size);
Box scale_box(const Box &box, const cv::Size &original_image_size,
              const cv::Size &resized_image_size);
yolo_msgs::msg::BoundingBox2D
convert_to_bounding_box(const yolo_onnx_utils::Box &box);

std::vector<yolo_onnx_utils::Box>
get_boxes(const std::vector<Ort::Value> &preds,
          const cv::Size &original_image_size,
          const cv::Size &resized_image_size, const int num_classes);

} // namespace yolo_onnx_utils

#endif // YOLO_CPP_ROS__YOLO__UTILS_HPP_