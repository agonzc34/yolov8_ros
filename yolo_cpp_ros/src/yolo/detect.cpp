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

#include "yolo_cpp_ros/yolo/detect.hpp"
#include "yolo_cpp_ros/yolo/utils.hpp"

namespace yolo_onnx {

YoloDetect::YoloDetect(yolo_onnx_utils::YoloParams params) : Model(params) {}

YoloDetect::~YoloDetect() {}

std::vector<yolo_msgs::msg::Detection>
YoloDetect::postprocess(const cv::Size &original_image_size,
                        const cv::Size &resized_image_size,
                        const std::vector<Ort::Value> &preds) {
  std::vector<yolo_msgs::msg::Detection> detection_array;
  std::vector<yolo_onnx_utils::Box> detections;

  std::vector<int64_t> shape = preds[0].GetTensorTypeAndShapeInfo().GetShape();

  if (shape.back() == 6) {
    // Process predictions without applying NMS
    detections = yolo_onnx_utils::get_detection_without_nms(
        preds, shape, original_image_size, resized_image_size);
  } else {
    // Process predictions applying NMS
    const size_t num_features = shape[1];
    const int num_classes = static_cast<int>(num_features) - 4;

    detections = yolo_onnx_utils::get_detection_with_nms(
        preds, shape, original_image_size, resized_image_size, num_classes, this->iou_threshold, this->conf_threshold);
  }

  for (size_t i = 0; i < detections.size(); ++i) {
    yolo_msgs::msg::Detection detection;
    detection.bbox = yolo_onnx_utils::convert_to_bounding_box(detections[i]);
    detection.score = detections[i].score;
    detection.class_id = detections[i].class_id;
    detection.id = "0";
    if (detections[i].class_id < static_cast<int>(this->class_names.size())) {
      detection.class_name = this->class_names[detections[i].class_id];
    } else {
      detection.class_name = "unknown";
    }
    detection_array.push_back(detection);
  }

  return detection_array;
}
} // namespace yolo_onnx