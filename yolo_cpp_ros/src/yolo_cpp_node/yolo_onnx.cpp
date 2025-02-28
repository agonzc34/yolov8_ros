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

#include "yolo_cpp_ros/yolo_onnx.hpp"

yolo_cpp_ros::YoloOnnx::YoloOnnx(const YoloParams &params,
                                 const std::vector<std::string> &classes) {}

yolo_cpp_ros::YoloOnnx::~YoloOnnx() {}

yolo_msgs::msg::DetectionArray
yolo_cpp_ros::YoloOnnx::detect(const sensor_msgs::msg::Image::SharedPtr &source,
                               bool verbose, bool stream, float conf, float iou,
                               std::pair<int, int> imgsz, bool half,
                               int max_det, bool augment, bool agnostic_nms,
                               bool retina_masks, const std::string &device) {
  return yolo_msgs::msg::DetectionArray();
}

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
void yolo_cpp_ros::YoloOnnx::set_classes(
    const std::vector<std::string> &classes) {}