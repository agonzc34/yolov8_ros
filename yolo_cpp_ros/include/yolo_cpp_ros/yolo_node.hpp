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

#ifndef YOLO_CPP_ROS__YOLO_NODE_HPP_
#define YOLO_CPP_ROS__YOLO_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include <memory>
#include <string>

#include "sensor_msgs/msg/image.hpp"
#include "yolo_cpp_ros/engine/model.hpp"
#include "yolo_msgs/msg/detection_array.hpp"

namespace yolo_rclcpp
{
class YoloNode : public rclcpp_lifecycle::LifecycleNode
{
public:
struct YoloParams
{
    std::string model_path;
    std::string device;
    float threshold;
    float iou;
    int image_reliability;
    std::string image_topic;
};

    YoloNode();

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State &);
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State &);
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State &);
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_cleanup(const rclcpp_lifecycle::State &);
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_shutdown(const rclcpp_lifecycle::State &);

protected:
    std::unique_ptr<yolo_onnx::Model> yolo_model;

    rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr detection_publisher;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription;

    void declare_params();
    YoloParams get_params();

    YoloParams yolo_params;
    bool params_declared = false;
    
    void create_yolo(std::string model_path);
    void destroy_yolo();

private:
    void recieve_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

};
}  // namespace yolo_rclcpp

#endif  // YOLO_RCLCPP__YOLO_NODE_HPP_