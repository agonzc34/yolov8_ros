#include "yolo_cpp_ros/engine/model.hpp"
#include <fstream>
#include <iostream>

yolo_onnx::Model::Model(std::string model_path) 
    : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8"),
      sessionOptions(),
      session(env, model_path.c_str(), sessionOptions),
      isDynamicInputShape(false)
{
    // Initialize session options
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Use CUDA provider if available
    try {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
        std::cout << "CUDA Execution Provider is used." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "CUDA Execution Provider is not available. Using default CPU provider." << std::endl;
    }

    // Get input and output node information
    numInputNodes = session.GetInputCount();
    numOutputNodes = session.GetOutputCount();

    // Allocate input and output node names
    for (size_t i = 0; i < numInputNodes; i++) {
        inputNodeNameAllocatedStrings.push_back(session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
        inputNames.push_back(inputNodeNameAllocatedStrings.back().get());
    }

    for (size_t i = 0; i < numOutputNodes; i++) {
        outputNodeNameAllocatedStrings.push_back(session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
        outputNames.push_back(outputNodeNameAllocatedStrings.back().get());
    }

    // Set input image shape (assuming static input shape for simplicity)
    inputImageShape = cv::Size(640, 640); // Example shape, should be set according to the model

    // Load class names from coco.names file
    std::ifstream classNamesFile("src/yolov8_ros/yolo_cpp_ros/conf/coco.names");
    if (!classNamesFile.is_open()) {
        std::cerr << "Error: Could not open coco.names file." << std::endl;
        return;
    }
    std::string line;
    while (std::getline(classNamesFile, line)) {
        classNames.push_back(line);
    }

    std::cout << "Model " << model_path << " has been successfully loaded." << std::endl;
}

yolo_onnx::Model::~Model() {}

yolo_msgs::msg::DetectionArray yolo_onnx::Model::detect(const sensor_msgs::msg::Image::SharedPtr &image)
{
    float* blob = nullptr;
    std::vector<int64_t> inputTensorShape; // Define a local variable for input tensor shape
    cv::Mat resizedImage = preprocess(image, blob, inputTensorShape);
    auto outputTensors = inference(resizedImage);
    return postprocess(cv::Size(image->width, image->height), inputImageShape, outputTensors, 0.5, 0.4);
}

yolo_msgs::msg::DetectionArray yolo_onnx::Model::track(const sensor_msgs::msg::Image::SharedPtr &image)
{
    return yolo_msgs::msg::DetectionArray();
}

cv::Mat yolo_onnx::Model::preprocess(const sensor_msgs::msg::Image::SharedPtr &image, float *&blob, std::vector<int64_t> &inputTensorShape)
{
    auto cv_image = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
    cv::Mat resizedImage;
    cv::resize(cv_image->image, resizedImage, inputImageShape);
    return resizedImage;
}

std::vector<Ort::Value> yolo_onnx::Model::inference(const cv::Mat &image)
{
    return std::vector<Ort::Value>();
}

yolo_msgs::msg::DetectionArray yolo_onnx::Model::postprocess(const cv::Size &originalImageSize, const cv::Size &resizedImageShape, const std::vector<Ort::Value> &outputTensors, float confThreshold, float iouThreshold)
{
    return yolo_msgs::msg::DetectionArray();
}
