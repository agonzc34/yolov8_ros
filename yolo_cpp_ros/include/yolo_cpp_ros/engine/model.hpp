// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__ENGINE__MODEL_HPP_
#define YOLO_CPP_ROS__ENGINE__MODEL_HPP_

#include <vector>
#include "yolo_msgs/msg/detection.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cv_bridge/cv_bridge.h>
#include "yolo_cpp_ros/yolo/utils.hpp"

namespace yolo_onnx {
class Model
{
public:
    Model(yolo_utils::YoloParams params);
    ~Model();

    std::vector<yolo_msgs::msg::Detection> detect(const cv::Mat &image);

    float conf_threshold{0.5};   // Confidence threshold for detections
    float iou_threshold{0.5};    // Intersection over union threshold for detections

protected:
    std::vector<std::string> class_names;            // Vector of class names loaded from file

private:
    void preprocess(const cv::Mat &image, std::vector<int64_t> &input_tensor_shape);
    std::vector<Ort::Value> inference(std::vector<int64_t> &input_tensor_shape);
    virtual std::vector<yolo_msgs::msg::Detection> postprocess(const cv::Size &original_image_size, const cv::Size &resized_image_size,
        const std::vector<Ort::Value> &preds);
    // Populate class_names from the ONNX graph metadata ("names"), falling
    // back to the coco.names file when the model carries no vocabulary.
    void load_class_names();

    Ort::Env env{nullptr};                         // ONNX Runtime environment
    Ort::SessionOptions session_options{nullptr};   // Session options for ONNX Runtime
    Ort::Session session{nullptr};                 // ONNX Runtime session for running inference
    cv::Size input_image_shape;                      // Expected input image shape for the model

    // Persistent input buffer (CHW, 1*3*H*W floats) reused every inference to
    // avoid re-allocating + copying the full blob per frame.
    std::vector<float> input_buffer_;

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