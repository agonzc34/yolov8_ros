// Copyright (C) 2026 Alejandro González Cantón
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef YOLO_CPP_ROS__TRACKING__KALMAN_FILTER_HPP_
#define YOLO_CPP_ROS__TRACKING__KALMAN_FILTER_HPP_

#include <array>
#include <utility>

namespace yolo_tracking {

using KalmanMean = std::array<double, 8>;   // x, y, a, h, vx, vy, va, vh
using KalmanCovariance = std::array<std::array<double, 8>, 8>;
using KalmanMeasurement = std::array<double, 4>;  // x, y, a, h (projected space)
using KalmanProjectedCov = std::array<std::array<double, 4>, 4>;

// KalmanFilterXYAH: 8-dimensional constant-velocity Kalman filter for HBB
// tracking in image space. State is (x, y, aspect, height, vx, vy, va, vh)
// and the observation model is linear (the box state is measured directly).
// Numerically equivalent to ultralytics' KalmanFilterXYAH (numpy port).
class KalmanFilterXYAH {
public:
  KalmanFilterXYAH() = default;

  std::pair<KalmanMean, KalmanCovariance> initiate(
      const KalmanMeasurement &measurement) const;

  std::pair<KalmanMean, KalmanCovariance> predict(
      const KalmanMean &mean, const KalmanCovariance &covariance) const;

  // (mean':4, cov':4x4) projection of the state distribution into
  // measurement (box) space.
  void project(const KalmanMean &mean, const KalmanCovariance &covariance,
               KalmanMeasurement &projected_mean,
               KalmanProjectedCov &projected_covariance) const;

  std::pair<KalmanMean, KalmanCovariance> update(
      const KalmanMean &mean, const KalmanCovariance &covariance,
      const KalmanMeasurement &measurement) const;

private:
  static constexpr double kStdWeightPosition = 1.0 / 20;
  static constexpr double kStdWeightVelocity = 1.0 / 160;
};

}  // namespace yolo_tracking

#endif  // YOLO_CPP_ROS__TRACKING__KALMAN_FILTER_HPP_
