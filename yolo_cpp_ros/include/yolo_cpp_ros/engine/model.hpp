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

namespace yolo_onnx
{
class Model
{
public:
    Model(std::string model_path);
    ~Model();

    std::vector<yolo_msgs::msg::Detection> detect(const cv::Mat &image);

    float confThreshold{0.5};   // Confidence threshold for detections
    float iouThreshold{0.5};    // Intersection over union threshold for detections

protected:
    std::vector<std::string> classNames;            // Vector of class names loaded from file

private:
    cv::Mat preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);
    std::vector<Ort::Value> inference(const cv::Mat &image, float *blob, std::vector<int64_t> &inputTensorShape);
    virtual std::vector<yolo_msgs::msg::Detection> postprocess(const cv::Size &originalImageSize, const cv::Size &resizedImageShape,
        const std::vector<Ort::Value> &preds);

    Ort::Env env{nullptr};                         // ONNX Runtime environment
    Ort::SessionOptions sessionOptions{nullptr};   // Session options for ONNX Runtime
    Ort::Session session{nullptr};                 // ONNX Runtime session for running inference
    bool isDynamicInputShape{};                    // Flag indicating if input shape is dynamic
    cv::Size inputImageShape;                      // Expected input image shape for the model

    // Vectors to hold allocated input and output node names
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char *> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char *> outputNames;

    size_t numInputNodes, numOutputNodes;           // Number of input and output nodes in the model
    Ort::MemoryInfo memoryInfo;                     // Memory information for ONNX Runtime
};
}

#endif  // YOLO_CPP_ROS__ENGINE__MODEL_HPP_