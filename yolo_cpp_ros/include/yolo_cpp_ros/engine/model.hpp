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

#ifndef YOLO_CPP_ROS__ENGINE__MODEL_HPP_
#define YOLO_CPP_ROS__ENGINE__MODEL_HPP_

#include <vector>
#include "yolo_msgs/msg/detection.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cv_bridge/cv_bridge.h>
#include "yolo_cpp_ros/yolo/utils.hpp"

namespace yolo_onnx
{
class Model
{
public:
    Model(yolo_onnx_utils::YoloParams params);
    ~Model();

    std::vector<yolo_msgs::msg::Detection> detect(const cv::Mat &image);

    float conf_threshold{0.5};   // Confidence threshold for detections
    float iou_threshold{0.5};    // Intersection over union threshold for detections

protected:
    std::vector<std::string> class_names;            // Vector of class names loaded from file

private:
    cv::Mat preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &input_tensor_shape);
    std::vector<Ort::Value> inference(const cv::Mat &image, float *blob, std::vector<int64_t> &input_tensor_shape);
    virtual std::vector<yolo_msgs::msg::Detection> postprocess(const cv::Size &original_image_size, const cv::Size &resized_image_size,
        const std::vector<Ort::Value> &preds);

    Ort::Env env{nullptr};                         // ONNX Runtime environment
    Ort::SessionOptions session_options{nullptr};   // Session options for ONNX Runtime
    Ort::Session session{nullptr};                 // ONNX Runtime session for running inference
    cv::Size input_image_shape;                      // Expected input image shape for the model

    // Vectors to hold allocated input and output node names
    std::vector<Ort::AllocatedStringPtr> input_node_name_alloc_strings;
    std::vector<const char *> inputNames;
    std::vector<Ort::AllocatedStringPtr> output_node_name_alloc_strings;
    std::vector<const char *> outputNames;

    size_t num_input_nodes, num_output_nodes;           // Number of input and output nodes in the model
    Ort::MemoryInfo memory_info;                     // Memory information for ONNX Runtime
};
}

#endif  // YOLO_CPP_ROS__ENGINE__MODEL_HPP_