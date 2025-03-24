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

#include "yolo_cpp_ros/engine/model.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include "yolo_cpp_ros/yolo/utils.hpp"

namespace yolo_onnx {
Model::Model(yolo_onnx_utils::YoloParams params)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolo"), session_options(),
      is_dynamic_input_shape(false), input_image_shape(), num_input_nodes(0),
      num_output_nodes(0), memory_info(nullptr) {
  // Initialize session options
  int n_threads = params.n_threads;
  this->conf_threshold = params.threshold;
  this->iou_threshold = params.iou;
  std::string model_path = params.model_path;

  if (n_threads == -1) {
    n_threads = std::thread::hardware_concurrency();
  }

  this->session_options.SetIntraOpNumThreads(n_threads);
  this->session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  auto providers = Ort::GetAvailableProviders();
  if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") !=
      providers.end()) {
    OrtSessionOptionsAppendExecutionProvider_CUDA(this->session_options, 0);
    std::cout << "CUDA Execution Provider is available and has been added."
              << std::endl;
  } else {
    std::cout << "CUDA Execution Provider is not available." << std::endl;
  }

  Ort::AllocatorWithDefaultOptions allocator;

  this->session =
      Ort::Session(this->env, model_path.c_str(), this->session_options);

  // Get input and output node information
  this->num_input_nodes = this->session.GetInputCount();
  this->num_output_nodes = this->session.GetOutputCount();

  // Allocate input and output node names
  for (size_t i = 0; i < this->num_input_nodes; i++) {
    this->input_node_name_alloc_strings.push_back(
        this->session.GetInputNameAllocated(i, allocator));
    this->inputNames.push_back(
        this->input_node_name_alloc_strings.back().get());
  }

  for (size_t i = 0; i < this->num_output_nodes; i++) {
    this->output_node_name_alloc_strings.push_back(
        this->session.GetOutputNameAllocated(i, allocator));
    this->outputNames.push_back(
        this->output_node_name_alloc_strings.back().get());
  }

  Ort::TypeInfo input_type_info = this->session.GetInputTypeInfo(0);
  std::vector<int64_t> input_tensor_shape_vec =
      input_type_info.GetTensorTypeAndShapeInfo().GetShape();

  if (input_tensor_shape_vec.size() >= 4) {
    this->input_image_shape = cv::Size(static_cast<int>(input_tensor_shape_vec[3]),
                                     static_cast<int>(input_tensor_shape_vec[2]));
    if (input_tensor_shape_vec[2] == -1 && input_tensor_shape_vec[3] == -1) {
      this->is_dynamic_input_shape = true;
    }
  } else {
    throw std::runtime_error("Invalid input tensor shape.");
  }

  // Load class names from coco.names file
  std::ifstream class_names_file("src/yolov8_ros/yolo_cpp_ros/conf/coco.names"); // TODO: change this path
  if (!class_names_file.is_open()) {
    std::cerr << "Error: Could not open coco.names file." << std::endl;
    return;
  }
  std::string line;
  while (std::getline(class_names_file, line)) {
    this->class_names.push_back(line);
  }

  this->memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::cout << "Model " << model_path << " has been successfully loaded."
            << std::endl;
}

Model::~Model() {}

std::vector<yolo_msgs::msg::Detection>
yolo_onnx::Model::detect(const cv::Mat &image) {
  float *blob_ptr = nullptr;

  std::vector<int64_t> input_tensor_shape = {1, 3, this->input_image_shape.height,
                                           this->input_image_shape.width};
  cv::Mat resized_image = preprocess(image, blob_ptr, input_tensor_shape);
  auto preds = inference(resized_image, blob_ptr, input_tensor_shape);
  delete[] blob_ptr;
  return postprocess(cv::Size(image.cols, image.rows), this->input_image_shape,
                     preds);
}

cv::Mat yolo_onnx::Model::preprocess(const cv::Mat &image, float *&blob,
                                     std::vector<int64_t> &input_tensor_shape) {
  cv::Mat resized_image =
    yolo_onnx_utils::letterbox(image, cv::Size(input_tensor_shape[3], input_tensor_shape[2]), cv::Scalar(114, 114, 114), this->is_dynamic_input_shape);
  resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);
  blob = new float[input_tensor_shape[1] * input_tensor_shape[2] *
                   input_tensor_shape[3]];

  std::vector<cv::Mat> chw(resized_image.channels());
  for (int i = 0; i < resized_image.channels(); ++i) {
    chw[i] = cv::Mat(resized_image.rows, resized_image.cols, CV_32FC1,
                     blob + i * resized_image.cols * resized_image.rows);
  }
  cv::split(resized_image, chw); // Split channels into the blob

  return resized_image;
}

std::vector<Ort::Value>
yolo_onnx::Model::inference(const cv::Mat &image, float *blob,
                            std::vector<int64_t> &input_tensor_shape) {
  size_t input_tensor_size =
      std::accumulate(input_tensor_shape.begin(), input_tensor_shape.end(), 1,
                      std::multiplies<int64_t>());
  std::vector<float> inputTensorValues(blob, blob + input_tensor_size);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      this->memory_info, inputTensorValues.data(), input_tensor_size,
      input_tensor_shape.data(), input_tensor_shape.size());

  std::vector<Ort::Value> predictions = this->session.Run(
      Ort::RunOptions{nullptr}, this->inputNames.data(), &input_tensor,
      this->num_input_nodes, this->outputNames.data(), this->num_output_nodes);

  return predictions;
}

std::vector<yolo_msgs::msg::Detection>
Model::postprocess(const cv::Size &original_image_size,
                   const cv::Size &resized_image_size,
                   const std::vector<Ort::Value> &preds) {
  return std::vector<yolo_msgs::msg::Detection>();
}

} // namespace yolo_onnx
