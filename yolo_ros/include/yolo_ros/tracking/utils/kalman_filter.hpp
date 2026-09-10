// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

/// @file
/// @brief 8-dimensional constant-velocity Kalman filter for ByteTrack.

#ifndef YOLO_ROS__TRACKING__UTILS__KALMAN_FILTER_HPP_
#define YOLO_ROS__TRACKING__UTILS__KALMAN_FILTER_HPP_

#include <array>
#include <utility>

/// @addtogroup yolo_tracking
/// @{
namespace yolo_ros::tracking::utils {

/// @brief Kalman state mean: x, y, a, h, vx, vy, va, vh.
using KalmanMean = std::array<double, 8>; // x, y, a, h, vx, vy, va, vh
/// @brief Kalman state covariance (8x8).
using KalmanCovariance = std::array<std::array<double, 8>, 8>;
/// @brief Measurement vector in projected (box) space: x, y, a, h.
using KalmanMeasurement = std::array<double, 4>; // x, y, a, h (projected space)
/// @brief Covariance projected into measurement space (4x4).
using KalmanProjectedCov = std::array<std::array<double, 4>, 4>;

/// @brief 8-dimensional constant-velocity Kalman filter for HBB tracking in
/// image space.
///
/// State is (x, y, aspect, height, vx, vy, va, vh) and the observation model
/// is linear (the box state is measured directly). Uses the model and
/// uncertainty weights from the original ByteTrack code.
class KalmanFilterXYAH {
public:
  /// @brief Construct a filter with the reference motion/uncertainty model.
  KalmanFilterXYAH() = default;

  /// @brief Initialize a state from a first measurement.
  /// @param[in] measurement Box measured as (x, y, aspect, height).
  /// @return The initial (mean, covariance) pair.
  std::pair<KalmanMean, KalmanCovariance>
  initiate(const KalmanMeasurement &measurement) const;

  /// @brief Predict the state one time step forward.
  /// @param[in] mean Current state mean.
  /// @param[in] covariance Current state covariance.
  /// @return The predicted (mean, covariance) pair.
  std::pair<KalmanMean, KalmanCovariance>
  predict(const KalmanMean &mean, const KalmanCovariance &covariance) const;

  /// @brief Project the state distribution into measurement (box) space.
  /// @param[in] mean Current state mean.
  /// @param[in] covariance Current state covariance.
  /// @param[out] projected_mean Mean in (x, y, a, h) space.
  /// @param[out] projected_covariance 4x4 covariance in projection space.
  void project(const KalmanMean &mean, const KalmanCovariance &covariance,
               KalmanMeasurement &projected_mean,
               KalmanProjectedCov &projected_covariance) const;

  /// @brief Correct the state with a new measurement.
  /// @param[in] mean Predicted state mean.
  /// @param[in] covariance Predicted state covariance.
  /// @param[in] measurement Observed box as (x, y, aspect, height).
  /// @return The updated (mean, covariance) pair.
  std::pair<KalmanMean, KalmanCovariance>
  update(const KalmanMean &mean, const KalmanCovariance &covariance,
         const KalmanMeasurement &measurement) const;

private:
  /// @brief Position uncertainty weight (1/20).
  static constexpr double kStdWeightPosition = 1.0 / 20;
  /// @brief Velocity uncertainty weight (1/160).
  static constexpr double kStdWeightVelocity = 1.0 / 160;
};

} // namespace yolo_ros::tracking::utils
/// @}

#endif // YOLO_ROS__TRACKING__UTILS__KALMAN_FILTER_HPP_
