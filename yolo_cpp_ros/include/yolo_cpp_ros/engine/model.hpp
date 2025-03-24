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

#ifndef YOLO_CPP_ROS__ENGINE__MODEL_HPP_
#define YOLO_CPP_ROS__ENGINE__MODEL_HPP_

#include <vector>
#include "yolo_msgs/msg/detection.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cv_bridge/cv_bridge.hpp>
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
    bool is_dynamic_input_shape{};                    // Flag indicating if input shape is dynamic
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