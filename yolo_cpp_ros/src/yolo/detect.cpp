#include "yolo_cpp_ros/yolo/detect.hpp"

namespace yolo_onnx {

YoloDetect::YoloDetect(std::string model_path) : Model(model_path) {}

YoloDetect::~YoloDetect() {}

yolo_msgs::msg::DetectionArray YoloDetect::postprocess(const cv::Size &originalImageSize,
                                                        const cv::Size &resizedImageShape,
                                                        const std::vector<Ort::Value> &outputTensors,
                                                        float confThreshold, float iouThreshold) {
  return yolo_msgs::msg::DetectionArray();
}
}  // namespace yolo_onnx