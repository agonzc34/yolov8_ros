// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

#ifndef YOLO_CPP_ROS__TRACKING__STRACK_HPP_
#define YOLO_CPP_ROS__TRACKING__STRACK_HPP_

#include <array>

#include "yolo_cpp_ros/tracking/utils/kalman_filter.hpp"

namespace yolo_tracking {

enum class TrackState { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

// Single-object track with the Kalman state and lifecycle bookkeeping used by
// the original MIT-licensed ByteTrack reference implementation.
class STrack {
public:
  // xywh: center-x, center-y, width, height. cls: class id. idx: the index of
  // this detection in the full detection array of the current frame.
  STrack(const std::array<float, 4> &xywh, float score, int class_id, int idx);

  // Predict the next Kalman state (mean/covariance) one step forward.
  void predict();

  // Activate a brand-new tracklet: assigns a track_id, initializes the Kalman
  // state from the measurement and sets bookkeeping fields.
  void activate(const KalmanFilterXYAH *kalman_filter, int frame_id);

  // Reactivate a previously lost track with a new detection.
  void re_activate(const STrack &new_track, int frame_id, bool new_id = false);

  // Refine an already-tracked object with its matched detection.
  void update(const STrack &new_track, int frame_id);

  void mark_lost() { state_ = TrackState::Lost; }
  void mark_removed() { state_ = TrackState::Removed; }
  int end_frame() const { return frame_id_; }

  // Next global track id.
  static int next_id();
  static void reset_id() { count_ = 0; }

  // --- accessors ----------------------------------------------------------
  int track_id() const { return track_id_; }
  int frame_id() const { return frame_id_; }
  int start_frame() const { return start_frame_; }
  int tracklet_len() const { return tracklet_len_; }
  TrackState state() const { return state_; }
  bool is_activated() const { return is_activated_; }
  float score() const { return score_; }
  int class_id() const { return class_id_; }
  int idx() const { return idx_; }

  // Current box in (min-x, min-y, max-x, max-y) pixel format.
  std::array<float, 4> xyxy() const;
  // Current box in (center-x, center-y, width, height) pixel format.
  std::array<float, 4> xywh() const;

  // top-left, width, height (both from the Kalman state).
  std::array<float, 4> tlwh() const;

  // tlwh -> (center-x, center-y, aspect = w / h, height).
  static std::array<double, 4> tlwh_to_xyah(const std::array<float, 4> &tlwh);

  static int count_;

private:
  std::array<float, 4> _tlwh_{};  // original detection box (tlwh)
  const KalmanFilterXYAH *kf_ = nullptr;
  KalmanMean mean_{};
  KalmanCovariance covariance_{};
  bool has_state_ = false;  // mean/covariance valid (activated at least once)

  bool is_activated_ = false;
  int track_id_ = 0;
  int frame_id_ = 0;
  int start_frame_ = 0;
  int tracklet_len_ = 0;
  TrackState state_ = TrackState::New;
  float score_ = 0.0f;
  int class_id_ = 0;
  int idx_ = -1;
};

}  // namespace yolo_tracking

#endif  // YOLO_CPP_ROS__TRACKING__STRACK_HPP_
