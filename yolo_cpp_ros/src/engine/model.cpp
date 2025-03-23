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

namespace yolo_onnx {

cv::Mat letterbox(const cv::Mat &img, const cv::Size &new_shape,
                  const cv::Scalar &color = cv::Scalar(114, 114, 114), bool auto_size = true) {
  int img_h = img.rows;
  int img_w = img.cols;

  // Compute scale factor and new dimensions
  float scale =
      std::min((float)new_shape.width / img_w, (float)new_shape.height / img_h);
  
  int new_w = std::round(img.cols * scale);
  int new_h = std::round(img.rows * scale);

  int new_pad_w = (new_shape.width - new_w) / 2;
  int new_pad_h = (new_shape.height - new_h) / 2;

  if (auto_size) {
    new_pad_w = new_pad_w % 32;
    new_pad_h = new_pad_h % 32;
  } else {
    new_w = new_shape.width;
    new_h = new_shape.height;
    scale = std::min((float)new_shape.width / img_w, (float)new_shape.height / img_h);
    new_pad_w = 0;
    new_pad_h = 0;
  }

  if (img.cols == new_w && img.rows == new_h) {
    return img;
  } else {
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    return resized_img; // TODO padded
  }
}

Model::Model(std::string model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolo"), sessionOptions(),
      isDynamicInputShape(false), inputImageShape(), numInputNodes(0),
      numOutputNodes(0), memoryInfo(nullptr) {
  // Initialize session options
  this->sessionOptions.SetIntraOpNumThreads(6); // TODO: Change
  this->sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  auto providers = Ort::GetAvailableProviders();
  if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") !=
      providers.end()) {
    OrtSessionOptionsAppendExecutionProvider_CUDA(this->sessionOptions, 0);
    std::cout << "CUDA Execution Provider is available and has been added."
              << std::endl;
  } else {
    std::cout << "CUDA Execution Provider is not available." << std::endl;
  }

  Ort::AllocatorWithDefaultOptions allocator;

  this->session =
      Ort::Session(this->env, model_path.c_str(), this->sessionOptions);

  // Get input and output node information
  this->numInputNodes = this->session.GetInputCount();
  this->numOutputNodes = this->session.GetOutputCount();

  // Allocate input and output node names
  for (size_t i = 0; i < this->numInputNodes; i++) {
    this->inputNodeNameAllocatedStrings.push_back(
        this->session.GetInputNameAllocated(i, allocator));
    this->inputNames.push_back(
        this->inputNodeNameAllocatedStrings.back().get());
  }

  for (size_t i = 0; i < this->numOutputNodes; i++) {
    this->outputNodeNameAllocatedStrings.push_back(
        this->session.GetOutputNameAllocated(i, allocator));
    this->outputNames.push_back(
        this->outputNodeNameAllocatedStrings.back().get());
  }

  Ort::TypeInfo inputTypeInfo = this->session.GetInputTypeInfo(0);
  std::vector<int64_t> inputTensorShapeVec =
      inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

  if (inputTensorShapeVec.size() >= 4) {
    this->inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]),
                                     static_cast<int>(inputTensorShapeVec[2]));
    if (inputTensorShapeVec[2] == -1 && inputTensorShapeVec[3] == -1) {
      this->isDynamicInputShape = true;
    }
  } else {
    throw std::runtime_error("Invalid input tensor shape.");
  }

  // Load class names from coco.names file
  std::ifstream classNamesFile("src/yolov8_ros/yolo_cpp_ros/conf/coco.names");
  if (!classNamesFile.is_open()) {
    std::cerr << "Error: Could not open coco.names file." << std::endl;
    return;
  }
  std::string line;
  while (std::getline(classNamesFile, line)) {
    this->classNames.push_back(line);
  }

  this->memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::cout << "Model " << model_path << " has been successfully loaded."
            << std::endl;
}

Model::~Model() {}

std::vector<yolo_msgs::msg::Detection>
yolo_onnx::Model::detect(const cv::Mat &image) {
  float *blobPtr = nullptr;

  std::vector<int64_t> inputTensorShape = {1, 3, this->inputImageShape.height,
                                           this->inputImageShape.width};
  cv::Mat resizedImage = preprocess(image, blobPtr, inputTensorShape);
  auto preds = inference(resizedImage, blobPtr, inputTensorShape);
  delete[] blobPtr;
  return postprocess(cv::Size(image.cols, image.rows), this->inputImageShape,
                     preds);
}

cv::Mat yolo_onnx::Model::preprocess(const cv::Mat &image, float *&blob,
                                     std::vector<int64_t> &inputTensorShape) {
  cv::Mat resizedImage =
      letterbox(image, cv::Size(inputTensorShape[3], inputTensorShape[2]), this->isDynamicInputShape);
  resizedImage.convertTo(resizedImage, CV_32FC3, 1.0 / 255.0);
  blob = new float[inputTensorShape[1] * inputTensorShape[2] *
                   inputTensorShape[3]];

  std::vector<cv::Mat> chw(resizedImage.channels());
  for (int i = 0; i < resizedImage.channels(); ++i) {
    chw[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1,
                     blob + i * resizedImage.cols * resizedImage.rows);
  }
  cv::split(resizedImage, chw); // Split channels into the blob

  return resizedImage;
}

std::vector<Ort::Value>
yolo_onnx::Model::inference(const cv::Mat &image, float *blob,
                            std::vector<int64_t> &inputTensorShape) {
  size_t inputTensorSize =
      std::accumulate(inputTensorShape.begin(), inputTensorShape.end(), 1,
                      std::multiplies<int64_t>());
  std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      this->memoryInfo, inputTensorValues.data(), inputTensorSize,
      inputTensorShape.data(), inputTensorShape.size());

  std::vector<Ort::Value> predictions = this->session.Run(
      Ort::RunOptions{nullptr}, this->inputNames.data(), &inputTensor,
      this->numInputNodes, this->outputNames.data(), this->numOutputNodes);

  return predictions;
}

std::vector<yolo_msgs::msg::Detection>
Model::postprocess(const cv::Size &originalImageSize,
                   const cv::Size &resizedImageShape,
                   const std::vector<Ort::Value> &preds) {
  return std::vector<yolo_msgs::msg::Detection>();
}

} // namespace yolo_onnx
