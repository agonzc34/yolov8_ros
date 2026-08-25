// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// Portions Copyright (c) 2012-2025 Tomas Kazmar
// SPDX-License-Identifier: MIT AND BSD-2-Clause

#ifndef YOLO_CPP_ROS__TRACKING__UTILS__LAPJV_HPP_
#define YOLO_CPP_ROS__TRACKING__UTILS__LAPJV_HPP_

#include <cstddef>
#include <vector>

namespace yolo_tracking {

// Dense Jonker-Volgenant solver for the linear assignment problem.
//
// Ported from ByteTrack's C++ deployment (MIT license):
//   deploy/{ncnn,TensorRT}/cpp/src/lapjv.cpp + include/lapjv.h
// which incorporates the BSD-2-Clause `lap` library's LAPJV implementation.
// See THIRD_PARTY_NOTICES.md.
//
//   n          : matrix order (cost must be n x n).
//   cost       : row-major cost matrix (vector of row vectors).
//   rowsol[i]  : column assigned to row i, or -1 if the solver's LARGE marker
//                (only used internally; callers manage unmatched entries).
//   colsol[j]  : row assigned to column j.
//
// Returns 0 on success, nonzero if allocation failed (very unlikely).
int lapjv_internal(std::size_t n, const std::vector<std::vector<double>> &cost,
                   std::vector<int> &rowsol, std::vector<int> &colsol);

}  // namespace yolo_tracking

#endif  // YOLO_CPP_ROS__TRACKING__UTILS__LAPJV_HPP_
