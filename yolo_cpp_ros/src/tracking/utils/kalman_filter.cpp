// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

#include "yolo_cpp_ros/tracking/utils/kalman_filter.hpp"

#include <cmath>
#include <cstddef>

namespace yolo_ros::tracking::utils {

namespace {

// 8x8 * 8x8 matrix product.
KalmanCovariance mul88(const KalmanCovariance &a, const KalmanCovariance &b) {
  KalmanCovariance out{};
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      double acc = 0.0;
      for (std::size_t k = 0; k < 8; ++k) {
        acc += a[i][k] * b[k][j];
      }
      out[i][j] = acc;
    }
  }
  return out;
}

// 8x8 * 8x1 vector product.
KalmanMean mul8v(const KalmanCovariance &a, const KalmanMean &v) {
  KalmanMean out{};
  for (std::size_t i = 0; i < 8; ++i) {
    double acc = 0.0;
    for (std::size_t k = 0; k < 8; ++k) {
      acc += a[i][k] * v[k];
    }
    out[i] = acc;
  }
  return out;
}

// 4x4 matrix inverse via Gauss-Jordan with full pivoting. Returns false if the
// matrix is numerically singular.
bool inv44(KalmanProjectedCov &m) {
  double a[4][8];
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      a[i][j] = m[i][j];
      a[i][j + 4] = (i == j) ? 1.0 : 0.0;
    }
  }
  for (std::size_t col = 0; col < 4; ++col) {
    // partial pivot
    std::size_t pivot = col;
    for (std::size_t r = col + 1; r < 4; ++r) {
      if (std::abs(a[r][col]) > std::abs(a[pivot][col])) {
        pivot = r;
      }
    }
    if (std::abs(a[pivot][col]) < 1e-12) {
      return false;
    }
    if (pivot != col) {
      for (std::size_t j = 0; j < 8; ++j) {
        std::swap(a[col][j], a[pivot][j]);
      }
    }
    const double inv = 1.0 / a[col][col];
    for (std::size_t j = 0; j < 8; ++j) {
      a[col][j] *= inv;
    }
    for (std::size_t r = 0; r < 4; ++r) {
      if (r == col) {
        continue;
      }
      const double factor = a[r][col];
      if (factor == 0.0) {
        continue;
      }
      for (std::size_t j = 0; j < 8; ++j) {
        a[r][j] -= factor * a[col][j];
      }
    }
  }
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      m[i][j] = a[i][j + 4];
    }
  }
  return true;
}

// 4x4 * 4x8 product: projected-cov-inverse times (C[:, :4])^T.
struct MatX {
  double d[4][8];
};

MatX mul48(const KalmanProjectedCov &a4,
           const std::array<std::array<double, 8>, 4> &b48) {
  MatX out{};
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      double acc = 0.0;
      for (std::size_t k = 0; k < 4; ++k) {
        acc += a4[i][k] * b48[k][j];
      }
      out.d[i][j] = acc;
    }
  }
  return out;
}

} // namespace

std::pair<KalmanMean, KalmanCovariance>
KalmanFilterXYAH::initiate(const KalmanMeasurement &measurement) const {
  KalmanMean mean{};
  for (std::size_t i = 0; i < 4; ++i) {
    mean[i] = measurement[i];
    mean[i + 4] = 0.0;
  }

  // The uncertainty weights follow the original ByteTrack reference constants
  // (2*std_position on position, 10*std_velocity on velocity).
  const double h = measurement[3];
  const double std[8] = {
      2 * kStdWeightPosition * h,
      2 * kStdWeightPosition * h,
      1e-2,
      2 * kStdWeightPosition * h,
      10 * kStdWeightVelocity * h,
      10 * kStdWeightVelocity * h,
      1e-5,
      10 * kStdWeightVelocity * h,
  };

  KalmanCovariance covariance{};
  for (std::size_t i = 0; i < 8; ++i) {
    covariance[i][i] = std[i] * std[i];
  }
  return {mean, covariance};
}

std::pair<KalmanMean, KalmanCovariance>
KalmanFilterXYAH::predict(const KalmanMean &mean,
                          const KalmanCovariance &covariance) const {
  // Constant-velocity motion model F = [I, dt*I; 0, I] with dt = 1.
  KalmanCovariance motion_mat{};
  for (std::size_t i = 0; i < 8; ++i) {
    motion_mat[i][i] = 1.0;
  }
  for (std::size_t i = 0; i < 4; ++i) {
    motion_mat[i][4 + i] = 1.0;
  }

  const double h = mean[3];
  const double std_pos[4] = {kStdWeightPosition * h, kStdWeightPosition * h,
                             1e-2, kStdWeightPosition * h};
  const double std_vel[4] = {kStdWeightVelocity * h, kStdWeightVelocity * h,
                             1e-5, kStdWeightVelocity * h};

  KalmanCovariance motion_cov{};
  for (std::size_t i = 0; i < 4; ++i) {
    motion_cov[i][i] = std_pos[i] * std_pos[i];
    motion_cov[4 + i][4 + i] = std_vel[i] * std_vel[i];
  }

  KalmanMean new_mean = mul8v(motion_mat, mean);
  // F * C * F^T  (motion_mat is symmetric in effect for this block structure,
  // but compute both products explicitly to stay generic).
  KalmanCovariance fc = mul88(motion_mat, covariance);
  KalmanCovariance fcf{};
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      double acc = 0.0;
      for (std::size_t k = 0; k < 8; ++k) {
        acc += fc[i][k] * motion_mat[j][k];
      }
      fcf[i][j] = acc;
    }
  }
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      fcf[i][j] += motion_cov[i][j];
    }
  }

  return {new_mean, fcf};
}

void KalmanFilterXYAH::project(const KalmanMean &mean,
                               const KalmanCovariance &covariance,
                               KalmanMeasurement &projected_mean,
                               KalmanProjectedCov &projected_covariance) const {
  const double h = mean[3];
  const double std[4] = {kStdWeightPosition * h, kStdWeightPosition * h, 1e-1,
                         kStdWeightPosition * h};
  for (std::size_t i = 0; i < 4; ++i) {
    projected_mean[i] = mean[i];
  }
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      projected_covariance[i][j] = covariance[i][j];
    }
    projected_covariance[i][i] += std[i] * std[i];
  }
}

std::pair<KalmanMean, KalmanCovariance>
KalmanFilterXYAH::update(const KalmanMean &mean,
                         const KalmanCovariance &covariance,
                         const KalmanMeasurement &measurement) const {
  KalmanMeasurement projected_mean{};
  KalmanProjectedCov projected_cov{};
  project(mean, covariance, projected_mean, projected_cov);

  // Kalman gain: K = S^{-1} (C H^T)^T  where H picks state[:4] and S is the
  // projected covariance. Equivalent to
  //   kalman_gain = solve(projected_cov, covariance[:, :4].T).T
  KalmanProjectedCov s_inv = projected_cov;
  inv44(s_inv);

  // B = (C[:, :4])^T  (4 rows, 8 cols): B[i][j] = covariance[j][i].
  std::array<std::array<double, 8>, 4> b{};
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      b[i][j] = covariance[j][i];
    }
  }
  MatX x = mul48(s_inv, b); // X (4x8) = S^{-1} * B
  // K (8x4):  K[j][i] = X[i][j]
  double k[8][4];
  for (std::size_t j = 0; j < 8; ++j) {
    for (std::size_t i = 0; i < 4; ++i) {
      k[j][i] = x.d[i][j];
    }
  }

  // innovation = measurement - projected_mean
  double innovation[4];
  for (std::size_t i = 0; i < 4; ++i) {
    innovation[i] = measurement[i] - projected_mean[i];
  }

  KalmanMean new_mean = mean;
  for (std::size_t j = 0; j < 8; ++j) {
    double acc = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
      acc += k[j][i] * innovation[i];
    }
    new_mean[j] += acc;
  }

  // new_cov = C - K S K^T
  // T (8x4) = K * S ; then R (8x8) = T * K^T
  double t[8][4] = {};
  for (std::size_t j = 0; j < 8; ++j) {
    for (std::size_t i = 0; i < 4; ++i) {
      double acc = 0.0;
      for (std::size_t l = 0; l < 4; ++l) {
        acc += k[j][l] * projected_cov[l][i];
      }
      t[j][i] = acc;
    }
  }
  KalmanCovariance new_covariance = covariance;
  for (std::size_t j1 = 0; j1 < 8; ++j1) {
    for (std::size_t j2 = 0; j2 < 8; ++j2) {
      double acc = 0.0;
      for (std::size_t i = 0; i < 4; ++i) {
        acc += t[j1][i] * k[j2][i];
      }
      new_covariance[j1][j2] -= acc;
    }
  }

  return {new_mean, new_covariance};
}

} // namespace yolo_ros::tracking::utils
