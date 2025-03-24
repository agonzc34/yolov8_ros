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
  std::vector<yolo_onnx_utils::Box> detections;

  if (preds[0].GetTensorTypeAndShapeInfo().GetShape().back() == 6) {
    // Process predictions without applying NMS

    std::vector<yolo_onnx_utils::Box> boxes;
    for (size_t i = 0; i < preds.size(); ++i) {
      auto pred = preds[i].GetTensorData<float>();
      for (size_t j = 0; j < preds[0].GetTensorTypeAndShapeInfo().GetShape()[0];
           ++j) {
        yolo_onnx_utils::Box box;
        box.x1 = pred[j * 6 + 0];
        box.y1 = pred[j * 6 + 1];
        box.x2 = pred[j * 6 + 2];
        box.y2 = pred[j * 6 + 3];
        box.score = pred[j * 6 + 4];
        box.class_id = static_cast<int>(pred[j * 6 + 5]);
        box.index = j;
        
        yolo_onnx_utils::Box scaled_box = yolo_onnx_utils::scale_box(box, originalImageSize, resizedImageShape);
        boxes.push_back(scaled_box);
      }
    }
    detections = boxes;
  } else {
    // Process predictions applying NMS
    std::vector<yolo_onnx_utils::Box> boxes;

    const float* rawOutput = preds[0].GetTensorData<float>(); // Extract raw output data from the first output tensor
    const std::vector<int64_t> outputShape = preds[0].GetTensorTypeAndShapeInfo().GetShape();
    const size_t num_features = outputShape[1];
    const size_t num_detections = outputShape[2];
    const int num_classes = static_cast<int>(num_features) - 4;

    const float* ptr = rawOutput;
    for (size_t i = 0; i < num_detections; ++i) {
      yolo_onnx_utils::Box box;
      float center_x = ptr[0 * num_detections + i];
      float center_y = ptr[1 * num_detections + i];
      float width = ptr[2 * num_detections + i];
      float height = ptr[3 * num_detections + i];

      int class_id = -1;
      float max_score = -1.0f;

      for (int j = 0; j < num_classes; ++j) {
        float score = ptr[(4 + j) * num_detections + i];
        if (score > max_score) {
          max_score = score;
          class_id = j;
        }
      }

      if (max_score > this->confThreshold) {
        box.x1 = (center_x - width / 2);
        box.y1 = (center_y - height / 2);
        box.x2 = (center_x + width / 2);
        box.y2 = (center_y + height / 2);
        box.score = max_score;
        box.class_id = class_id;

        yolo_onnx_utils::Box scaled_box = yolo_onnx_utils::scale_box(box, originalImageSize, resizedImageShape);
        boxes.push_back(scaled_box);
      }
    }

    detections = yolo_onnx_utils::nms(boxes, this->iouThreshold, this->confThreshold);
  }

  for (size_t i = 0; i < detections.size(); ++i) {
    yolo_msgs::msg::Detection detection;
    detection.bbox.center.position.x = (detections[i].x1 + detections[i].x2) / 2;
    detection.bbox.center.position.y = (detections[i].y1 + detections[i].y2) / 2;
    detection.bbox.size.x = detections[i].x2 - detections[i].x1;
    detection.bbox.size.y = detections[i].y2 - detections[i].y1;
    detection.score = detections[i].score;
    detection.class_id = detections[i].class_id;
    detection.id = "0";
    detection.class_name = this->classNames[detections[i].class_id];
    detection_array.push_back(detection);
  }

  return detection_array;
}
} // namespace yolo_onnx