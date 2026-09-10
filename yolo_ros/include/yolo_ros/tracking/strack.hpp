// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

/// @file
/// @brief Single-object track (STrack) with Kalman state and lifecycle
/// bookkeeping for the ByteTrack implementation.

#ifndef YOLO_ROS__TRACKING__STRACK_HPP_
#define YOLO_ROS__TRACKING__STRACK_HPP_

#include <array>

#include "yolo_ros/tracking/utils/kalman_filter.hpp"

/// @addtogroup yolo_tracking
/// @{
namespace yolo_ros::tracking {

/// @brief Lifecycle state of a track.
enum class TrackState {
  New = 0,     ///< Just created, not yet fully activated.
  Tracked = 1, ///< Currently matched and active.
  Lost = 2,    ///< Not matched this frame but kept for re-association.
  Removed = 3  ///< Removed from the active/lost pools.
};

/// @brief Single-object track with the Kalman state and lifecycle bookkeeping
/// used by the original MIT-licensed ByteTrack reference implementation.
class STrack {
public:
  /// @brief Create a track from a detection box.
  /// @param xywh Box as (center-x, center-y, width, height).
  /// @param score Detection confidence in [0, 1].
  /// @param class_id Zero-based class id.
  /// @param idx Index of this detection in the full detection array of the
  /// current frame.
  STrack(const std::array<float, 4> &xywh, float score, int class_id, int idx);

  /// @brief Predict the next Kalman state (mean/covariance) one step forward.
  void predict();

  /// @brief Activate a brand-new tracklet.
  ///
  /// Assigns a track_id, initializes the Kalman state from the measurement and
  /// sets bookkeeping fields.
  /// @param[in] kalman_filter Filter used to initiate the state.
  /// @param[in] frame_id Current frame index.
  void activate(const utils::KalmanFilterXYAH *kalman_filter, int frame_id);

  /// @brief Reactivate a previously lost track with a new detection.
  /// @param[in] new_track Detection to re-associate.
  /// @param[in] frame_id Current frame index.
  /// @param[in] new_id When true, assign a fresh track id instead of reusing
  /// this track's id.
  void re_activate(const STrack &new_track, int frame_id, bool new_id = false);

  /// @brief Refine an already-tracked object with its matched detection.
  /// @param[in] new_track Matched detection.
  /// @param[in] frame_id Current frame index.
  void update(const STrack &new_track, int frame_id);

  /// @brief Mark the track as lost.
  void mark_lost() { state_ = TrackState::Lost; }
  /// @brief Mark the track as removed.
  void mark_removed() { state_ = TrackState::Removed; }
  /// @brief Frame index at which the track last ended.
  /// @return The internal frame_id_.
  int end_frame() const { return frame_id_; }

  /// @brief Return and advance the global track-id counter.
  /// @return A fresh track id.
  static int next_id();
  /// @brief Reset the global track-id counter to zero.
  static void reset_id() { count_ = 0; }

  // --- accessors ----------------------------------------------------------
  /// @brief Stable track id. @return The id.
  int track_id() const { return track_id_; }
  /// @brief Current frame index. @return The frame id.
  int frame_id() const { return frame_id_; }
  /// @brief Frame at which the track started. @return The start frame.
  int start_frame() const { return start_frame_; }
  /// @brief Number of frames the tracklet has existed. @return The length.
  int tracklet_len() const { return tracklet_len_; }
  /// @brief Current lifecycle state. @return The state.
  TrackState state() const { return state_; }
  /// @brief Whether the track is activated. @return True when activated.
  bool is_activated() const { return is_activated_; }
  /// @brief Detection confidence. @return The score.
  float score() const { return score_; }
  /// @brief Zero-based class id. @return The class id.
  int class_id() const { return class_id_; }
  /// @brief Detection index in the current frame. @return The index.
  int idx() const { return idx_; }

  /// @brief Current box in (min-x, min-y, max-x, max-y) pixel format.
  /// @return The xyxy corners.
  std::array<float, 4> xyxy() const;
  /// @brief Current box in (center-x, center-y, width, height) pixel format.
  /// @return The xywh box.
  std::array<float, 4> xywh() const;

  /// @brief Box as top-left x, top-left y, width, height (both from the Kalman
  /// state).
  /// @return The tlwh box.
  std::array<float, 4> tlwh() const;

  /// @brief Convert tlwh to (center-x, center-y, aspect = w / h, height).
  /// @param[in] tlwh Box as top-left x, top-left y, width, height.
  /// @return The xyah box.
  static std::array<double, 4> tlwh_to_xyah(const std::array<float, 4> &tlwh);

  /// @brief Global track-id counter shared by all STrack instances.
  static int count_;

private:
  /// @brief Original detection box (top-left x, top-left y, width, height).
  std::array<float, 4> _tlwh_{}; // original detection box (tlwh)
  /// @brief Kalman filter used for predict/update (non-owning).
  const utils::KalmanFilterXYAH *kf_ = nullptr;
  /// @brief Kalman state mean (x, y, a, h, vx, vy, va, vh).
  utils::KalmanMean mean_{};
  /// @brief Kalman state covariance (8x8).
  utils::KalmanCovariance covariance_{};
  /// @brief Whether mean_/covariance_ are valid (activated at least once).
  bool has_state_ = false; // mean/covariance valid (activated at least once)

  /// @brief Whether the track is currently activated.
  bool is_activated_ = false;
  /// @brief Stable track id.
  int track_id_ = 0;
  /// @brief Current frame index.
  int frame_id_ = 0;
  /// @brief Frame at which the track started.
  int start_frame_ = 0;
  /// @brief Number of frames the tracklet has existed.
  int tracklet_len_ = 0;
  /// @brief Current lifecycle state.
  TrackState state_ = TrackState::New;
  /// @brief Latest detection confidence.
  float score_ = 0.0f;
  /// @brief Zero-based class id.
  int class_id_ = 0;
  /// @brief Detection index in the current frame (-1 when unset).
  int idx_ = -1;
};

} // namespace yolo_ros::tracking
/// @}

#endif // YOLO_ROS__TRACKING__STRACK_HPP_
