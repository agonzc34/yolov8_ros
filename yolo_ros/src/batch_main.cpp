// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <memory>
#include <rclcpp/executors.hpp>

#include "yolo_ros/node/batch_node.hpp"

using namespace yolo_ros::node;

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<BatchNode>();
  node->configure();
  node->activate();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());

  executor.spin();

  executor.remove_node(node->get_node_base_interface());
  node.reset();

  rclcpp::shutdown();

  return 0;
}
