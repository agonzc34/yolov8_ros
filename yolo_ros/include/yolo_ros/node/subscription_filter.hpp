// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief message_filters input that owns an rclcpp subscription.
///
/// Galactic's message_filters::Subscriber only accepts rclcpp::Node*
/// (rclcpp_lifecycle::LifecycleNode is not an rclcpp::Node before Humble),
/// which makes it unusable from a lifecycle node. This adapter creates the
/// subscription through the free rclcpp::create_subscription() API (which works
/// with any node type) and feeds the message into the message_filters graph, so
/// it can be connected to a message_filters::Synchronizer exactly like
/// message_filters::Subscriber.

#ifndef YOLO_ROS__NODE__SUBSCRIPTION_FILTER_HPP_
#define YOLO_ROS__NODE__SUBSCRIPTION_FILTER_HPP_

#include <message_filters/simple_filter.h>
#include <rclcpp/rclcpp.hpp>

#include <string>

/// @addtogroup yolo_nodes
/// @{
namespace yolo_ros::node {

/// @brief message_filters source backed by an rclcpp subscription on @p NodeT
/// (rclcpp::Node or rclcpp_lifecycle::LifecycleNode).
/// @tparam M Message type (e.g. sensor_msgs::msg::Image).
template <typename M>
class NodeSubscription : public message_filters::SimpleFilter<M> {
public:
  /// @brief Create the subscription on @p node and feed the filter.
  /// @param[in] node Node owning the subscription (any node type).
  /// @param[in] topic Topic to subscribe to.
  /// @param[in] qos QoS profile.
  template <typename NodeT>
  void subscribe(NodeT &node, const std::string &topic,
                 const rclcpp::QoS &qos) {
    // The callback takes the message by value so rclcpp deduces
    // CallbackMessageT as M (a by-reference ConstSharedPtr is not unwrapped on
    // Galactic).
    subscription_ = rclcpp::create_subscription<M>(
        node, topic, qos,
        [this](typename M::ConstSharedPtr msg) { this->signalMessage(msg); });
  }

  /// @brief Drop the underlying subscription.
  void unsubscribe() { subscription_.reset(); }

private:
  /// @brief Subscription created by subscribe().
  typename rclcpp::Subscription<M>::SharedPtr subscription_;
};

} // namespace yolo_ros::node
/// @}

#endif // YOLO_ROS__NODE__SUBSCRIPTION_FILTER_HPP_
