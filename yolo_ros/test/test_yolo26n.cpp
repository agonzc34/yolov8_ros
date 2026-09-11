// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "test_model_helpers.hpp"
#include "yolo_ros/yolo/classify.hpp"
#include "yolo_ros/yolo/detect.hpp"
#include "yolo_ros/yolo/pose.hpp"
#include "yolo_ros/yolo/segment.hpp"

namespace yolo_ros::yolo {
namespace {

constexpr char kRepo[] = "zwh20081/yolo26-onnx";

class ExposedDetect : public YoloDetect {
public:
  using YoloDetect::YoloDetect;
  std::size_t num_class_names() const { return class_names.size(); }
};

class Detect26nTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string model =
        yolo_ros::test::download_model(kRepo, "yolo26n.onnx");
    image_ = yolo_ros::test::download_sample_image();
    if (model.empty() || image_.empty()) {
      GTEST_SKIP() << "yolo26n.onnx or the sample image is unavailable";
    }
    model_path_ = model;
    detector_ =
        std::make_unique<ExposedDetect>(yolo_ros::test::make_params(model));
  }

  cv::Mat image_;
  std::string model_path_;
  std::unique_ptr<ExposedDetect> detector_;
};

TEST_F(Detect26nTest, LoadsCocoVocabulary) {
  EXPECT_EQ(detector_->num_class_names(), 80u);
}

TEST_F(Detect26nTest, ProducesWellFormedDetections) {
  const auto dets = detector_->detect(image_);
  ASSERT_FALSE(dets.empty());
  for (const auto &d : dets) {
    EXPECT_GT(d.score, 0.0f);
    EXPECT_LE(d.score, 1.0f);
    EXPECT_GE(d.class_id, 0);
    EXPECT_LT(d.class_id, 80);
    EXPECT_FALSE(d.class_name.empty());
    EXPECT_NE(d.class_name, "unknown");
    EXPECT_EQ(d.id, "0");
    EXPECT_GE(d.bbox.center.position.x - d.bbox.size.x / 2.0, -1.0);
    EXPECT_GE(d.bbox.center.position.y - d.bbox.size.y / 2.0, -1.0);
    EXPECT_LE(d.bbox.center.position.x + d.bbox.size.x / 2.0,
              static_cast<double>(image_.cols) + 1.0);
    EXPECT_LE(d.bbox.center.position.y + d.bbox.size.y / 2.0,
              static_cast<double>(image_.rows) + 1.0);
  }
}

TEST_F(Detect26nTest, HigherThresholdFiltersLowScores) {
  const auto low = detector_->detect(image_);
  ASSERT_FALSE(low.empty());

  auto strict_params = yolo_ros::test::make_params(model_path_);
  strict_params.threshold = 0.5f;
  ExposedDetect strict(strict_params);
  const auto high = strict.detect(image_);

  ASSERT_LE(high.size(), low.size());
  for (const auto &d : high) {
    EXPECT_GE(d.score, 0.5f);
  }
  const bool any_above = std::any_of(
      low.begin(), low.end(),
      [](const yolo_msgs::msg::Detection &d) { return d.score >= 0.5f; });
  if (any_above) {
    EXPECT_FALSE(high.empty());
  }
}

class Segment26nTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string model =
        yolo_ros::test::download_model(kRepo, "yolo26n-seg.onnx");
    image_ = yolo_ros::test::download_sample_image();
    if (model.empty() || image_.empty()) {
      GTEST_SKIP() << "yolo26n-seg.onnx or the sample image is unavailable";
    }
    segment_ =
        std::make_unique<YoloSegment>(yolo_ros::test::make_params(model));
  }
  cv::Mat image_;
  std::unique_ptr<YoloSegment> segment_;
};

TEST_F(Segment26nTest, MasksMatchImageSize) {
  const auto dets = segment_->detect(image_);
  ASSERT_FALSE(dets.empty());
  bool any_nonempty = false;
  for (const auto &d : dets) {
    EXPECT_EQ(d.mask.width, static_cast<int32_t>(image_.cols));
    EXPECT_EQ(d.mask.height, static_cast<int32_t>(image_.rows));
    if (!d.mask.data.empty()) {
      any_nonempty = true;
      EXPECT_GE(d.mask.data.size(), 3u);
    }
    for (const auto &p : d.mask.data) {
      EXPECT_GE(p.x, 0);
      EXPECT_GE(p.y, 0);
      EXPECT_LT(p.x, image_.cols);
      EXPECT_LT(p.y, image_.rows);
    }
  }
  EXPECT_TRUE(any_nonempty);
}

class Pose26nTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string model =
        yolo_ros::test::download_model(kRepo, "yolo26n-pose.onnx");
    image_ = yolo_ros::test::download_people_image();
    if (model.empty() || image_.empty()) {
      GTEST_SKIP() << "yolo26n-pose.onnx or the people image is unavailable";
    }
    pose_ = std::make_unique<YoloPose>(yolo_ros::test::make_params(model));
  }
  cv::Mat image_;
  std::unique_ptr<YoloPose> pose_;
};

TEST_F(Pose26nTest, KeypointsAreWellFormed) {
  const auto dets = pose_->detect(image_);
  ASSERT_FALSE(dets.empty());
  bool any_keypoints = false;
  for (const auto &d : dets) {
    for (const auto &kp : d.keypoints.data) {
      any_keypoints = true;
      EXPECT_GE(kp.id, 1);
      EXPECT_LE(kp.id, 17);
      EXPECT_GE(kp.score, 0.0f);
      EXPECT_LE(kp.score, 1.0f);
      EXPECT_GE(kp.point.x, 0.0);
      EXPECT_LE(kp.point.x, image_.cols);
      EXPECT_GE(kp.point.y, 0.0);
      EXPECT_LE(kp.point.y, image_.rows);
    }
  }
  EXPECT_TRUE(any_keypoints);
}

class ExposedClassify : public YoloClassify {
public:
  using YoloClassify::YoloClassify;
  std::size_t num_class_names() const { return class_names.size(); }
};

class Classify26nTest : public ::testing::Test {
protected:
  void SetUp() override {
    model_path_ = yolo_ros::test::download_model(kRepo, "yolo26n-cls.onnx");
    image_ = yolo_ros::test::download_sample_image();
    if (model_path_.empty() || image_.empty()) {
      GTEST_SKIP() << "yolo26n-cls.onnx or the sample image is unavailable";
    }
    classify_ = std::make_unique<ExposedClassify>(
        yolo_ros::test::make_params(model_path_));
  }
  std::string model_path_;
  cv::Mat image_;
  std::unique_ptr<ExposedClassify> classify_;
};

TEST_F(Classify26nTest, LoadsImageNetVocabulary) {
  EXPECT_EQ(classify_->num_class_names(), 1000u);
}

TEST_F(Classify26nTest, TopKIsAProbabilityDistribution) {
  const auto dets = classify_->detect(image_);
  ASSERT_EQ(dets.size(), 5u);
  float sum = 0.0f;
  for (std::size_t i = 0; i < dets.size(); ++i) {
    EXPECT_GE(dets[i].score, 0.0f);
    EXPECT_LE(dets[i].score, 1.0f);
    EXPECT_GE(dets[i].class_id, 0);
    EXPECT_LT(dets[i].class_id, 1000);
    EXPECT_FALSE(dets[i].class_name.empty());
    EXPECT_NE(dets[i].class_name, "unknown");
    EXPECT_EQ(dets[i].bbox.size.x, 0.0);
    EXPECT_EQ(dets[i].bbox.size.y, 0.0);
    if (i > 0) {
      EXPECT_GE(dets[i - 1].score, dets[i].score);
    }
    sum += dets[i].score;
  }
  EXPECT_LE(sum, 1.0f + 1e-2f);
  // The graph bakes in a softmax, so these are 5 of 1000 probabilities: their
  // sum is well below 1. A double-softmax would flatten the row toward
  // uniform (1/1000 ~ 0.001), so the top class staying above 0.01 guards the
  // documented regression without requiring a specific model confidence.
  EXPECT_GT(dets.front().score, 0.01f);
}

TEST_F(Classify26nTest, TopKParameterIsHonored) {
  auto params = yolo_ros::test::make_params(model_path_);
  params.top_k = 3;
  ExposedClassify top3(params);
  EXPECT_EQ(top3.detect(image_).size(), 3u);
}

} // namespace
} // namespace yolo_ros::yolo
