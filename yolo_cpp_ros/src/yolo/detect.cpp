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

YoloDetect::YoloDetect(std::string model_path) : Model(model_path) {}

YoloDetect::~YoloDetect() {}

std::vector<yolo_msgs::msg::Detection>
YoloDetect::postprocess(const cv::Size &originalImageSize,
                        const cv::Size &resizedImageShape,
                        const std::vector<Ort::Value> &preds) {
  std::vector<yolo_msgs::msg::Detection> detection_array;

  if (preds[0].GetTensorTypeAndShapeInfo().GetShape().back() == 6) {
    // Process predictions without applying NMS
    for (size_t i = 0; i < preds.size(); ++i) {
      auto pred = preds[i].GetTensorData<float>();
      for (size_t j = 0; j < preds[0].GetTensorTypeAndShapeInfo().GetShape()[0];
           ++j) {
        float confidence = pred[j * 6 + 4];
        if (confidence > confThreshold) {
          yolo_msgs::msg::Detection detection;
          detection.bbox.center.position.x = (pred[j * 6 + 0] + pred[j * 6 + 2]) / 2;
          detection.bbox.center.position.y = (pred[j * 6 + 1] + pred[j * 6 + 3]) / 2;
          detection.bbox.size.x = pred[j * 6 + 2] - pred[j * 6 + 0];
          detection.bbox.size.y = pred[j * 6 + 3] - pred[j * 6 + 1];
          detection.score = confidence;
          detection.class_id = static_cast<int>(pred[j * 6 + 5]);

          detection_array.push_back(detection);
        }
      }
    }
  } else {
    // Process predictions applying NMS
    std::vector<yolo_onnx_utils::Box> boxes;
    for (size_t i = 0; i < preds.size(); ++i) {
      auto pred = preds[i].GetTensorData<float>();
      for (size_t j = 0; j < preds[0].GetTensorTypeAndShapeInfo().GetShape()[0];
           ++j) {
        float confidence = pred[j * 6 + 4];
        if (confidence > confThreshold) {
          yolo_onnx_utils::Box box;
          box.x1 = pred[j * 6 + 0];
          box.y1 = pred[j * 6 + 1];
          box.x2 = pred[j * 6 + 2];
          box.y2 = pred[j * 6 + 3];
          box.score = confidence;
          box.class_id = static_cast<int>(pred[j * 6 + 5]);
          box.index = j;
          boxes.push_back(box);
        }
      }
    }

    auto detections = yolo_onnx_utils::nms(boxes, iouThreshold, confThreshold);
    for (const auto &detection : detections) {
      yolo_msgs::msg::Detection detection_msg;
      detection_msg.bbox.center.position.x = (detection.x1 + detection.x2) / 2;
      detection_msg.bbox.center.position.y = (detection.y1 + detection.y2) / 2;
      detection_msg.bbox.size.x = detection.x2 - detection.x1;
      detection_msg.bbox.size.y = detection.y2 - detection.y1;
      detection_msg.score = detection.score;
      detection_msg.class_id = detection.class_id;

      detection_array.push_back(detection_msg);
    }
  }

  return detection_array;
}
} // namespace yolo_onnx