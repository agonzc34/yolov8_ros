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

#ifndef YOLO_CPP_ROS__YOLO_ONNX_HPP_
#define YOLO_CPP_ROS__YOLO_ONNX_HPP_

#include "yolo_msgs/msg/detection_array.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <memory>
#include <string>

namespace yolo_cpp_ros {
struct YoloParams {
  std::string model;
  std::string device;
  float threshold;
  float iou;
  int imgsz_height;
  int imgsz_width;
  bool half;
  int max_det;
  bool augment;
  bool agnostic_nms;
  bool retina_masks;
};

class YoloOnnx {
public:
  YoloOnnx(const YoloParams &params, const std::vector<std::string> &classes);
  virtual ~YoloOnnx();

  yolo_msgs::msg::DetectionArray
  detect(const sensor_msgs::msg::Image::SharedPtr &source, bool verbose = false,
         bool stream = false, float conf = 0.5, float iou = 0.5,
         std::pair<int, int> imgsz = {640, 640}, bool half = false,
         int max_det = 1000, bool augment = false, bool agnostic_nms = false,
         bool retina_masks = false, const std::string &device = "cpu");

  void set_classes(const std::vector<std::string> &classes);

protected:
    YoloParams params_;
    std::vector<std::string> classes_;
};
} // namespace yolo_cpp_ros

#endif // YOLO_CPP_ROS__YOLO_ONNX_HPP_