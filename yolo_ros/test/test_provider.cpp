// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "yolo_ros/engine/provider.hpp"

namespace yolo_ros::engine {
namespace {

TEST(ParseDeviceId, ParsesOrdinals) {
  EXPECT_EQ(parse_device_id("cuda:0"), 0);
  EXPECT_EQ(parse_device_id("trt:1"), 1);
  EXPECT_EQ(parse_device_id("tensorrt:2"), 2);
  EXPECT_EQ(parse_device_id("0"), 0);
  EXPECT_EQ(parse_device_id("3"), 3);
}

TEST(ParseDeviceId, DefaultsToZero) {
  EXPECT_EQ(parse_device_id("cuda"), 0);
  EXPECT_EQ(parse_device_id("cpu"), 0);
  EXPECT_EQ(parse_device_id(""), 0);
  EXPECT_EQ(parse_device_id("cuda:x"), 0);
  EXPECT_EQ(parse_device_id("cuda:-1"), 0);
}

TEST(ParseDeviceId, RejectsTrailingGarbage) {
  EXPECT_EQ(parse_device_id("cuda:1.5"), 0);
  EXPECT_EQ(parse_device_id("cuda:2abc"), 0);
  EXPECT_EQ(parse_device_id("cuda: 1"), 0);
}

TEST(TensorrtAvailable, DoesNotThrow) {
  const bool available = tensorrt_available();
  (void)available; // true on the robot build; the call must be safe either way
}

} // namespace
} // namespace yolo_ros::engine
