// MIT License
//
// Copyright (c) 2025 Alejandro González Cantón
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef YOLO_CPP_ROS__YOLO__UTILS_HPP_
#define YOLO_CPP_ROS__YOLO__UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <vector>

namespace yolo_onnx_utils {
struct Box {
  float x1, y1, x2, y2, score;
  int index, class_id;
};

float iou(const Box &box1, const Box &box2);
std::vector<struct Box> nms(std::vector<Box> &boxes, float iouThreshold,
                            float confThreshold);
cv::Mat letterbox(const cv::Mat &img, const cv::Size &new_shape,
                  const cv::Scalar &color,
                  bool auto_size);
Box scale_box(const Box &box, const cv::Size &originalImageSize,
              const cv::Size &resizedImageShape);
} // namespace yolo_onnx_utils

#endif // YOLO_CPP_ROS__YOLO__UTILS_HPP_