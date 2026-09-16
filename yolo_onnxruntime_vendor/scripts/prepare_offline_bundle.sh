#!/usr/bin/env bash
# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT
#
# Run on an INTERNET-CONNECTED host to produce a self-contained bundle for
# building ONNX Runtime 1.6.0 (aarch64, CUDA + TensorRT) on the offline robot.
#
# The bundle contains:
#   - onnxruntime/                 full v1.6.0 source tree with all submodules
#                                  (git metadata stripped; the robot build needs
#                                  no network)
#   - models/                      the opset-12 ONNX models for the pipeline
#
# Usage: scripts/prepare_offline_bundle.sh [output_dir]
#   output_dir defaults to <package>/offline-bundle; the tarball is
#   <output_dir>.tar.gz
#
# Env overrides:
#   MODELS_DIR  directory holding the .onnx files (default /home/agonzc34/models)
#   MODELS      space-separated model file names to include
set -euo pipefail

ORT_VERSION="1.6.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/../offline-bundle}"
MODELS_DIR="${MODELS_DIR:-/home/agonzc34/models}"
MODELS="${MODELS:-yolo26m.onnx yolo11n-seg.onnx yolo11n-segment.onnx yolo11n-pose.onnx yolo26m-pose.onnx}"
ORT_GIT_URL="https://github.com/microsoft/onnxruntime"

for tool in git tar; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "error: ${tool} not found" >&2
    exit 1
  }
done

echo "==> Cloning ONNX Runtime v${ORT_VERSION} with submodules (shallow)..."
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
git clone --depth 1 --shallow-submodules --recursive \
  -b "v${ORT_VERSION}" "${ORT_GIT_URL}" "${OUT_DIR}/onnxruntime"

echo "==> Verifying required submodules are populated..."
for sub in cmake/external/protobuf cmake/external/onnx cmake/external/onnx-tensorrt \
  cmake/external/eigen cmake/external/flatbuffers cmake/external/nsync; do
  if [[ ! -e "${OUT_DIR}/onnxruntime/${sub}" ]]; then
    echo "error: submodule ${sub} is missing" >&2
    exit 1
  fi
done

echo "==> Stripping git metadata (offline builds call no git)..."
git -C "${OUT_DIR}/onnxruntime" submodule foreach --recursive 'rm -rf .git' || true
rm -rf "${OUT_DIR}/onnxruntime/.git"

echo "==> Collecting opset-12 models from ${MODELS_DIR}..."
mkdir -p "${OUT_DIR}/models"
found=0
for model in ${MODELS}; do
  if [[ -f "${MODELS_DIR}/${model}" ]]; then
    cp -v "${MODELS_DIR}/${model}" "${OUT_DIR}/models/"
    found=$((found + 1))
  else
    echo "warning: ${MODELS_DIR}/${model} not found; skipping" >&2
  fi
done
if [[ "${found}" -eq 0 ]]; then
  echo "error: no models found in ${MODELS_DIR}" >&2
  exit 1
fi

BUNDLE="${OUT_DIR}.tar.gz"
echo "==> Creating ${BUNDLE}..."
tar -C "$(dirname "${OUT_DIR}")" -czf "${BUNDLE}" "$(basename "${OUT_DIR}")"

SIZE="$(du -h "${BUNDLE}" | cut -f1)"
cat <<EOF

Bundle ready: ${BUNDLE} (${SIZE})

Copy it to the robot (replace <user>@<robot>):
  scp "${BUNDLE}" <user>@<robot>:~/

On the robot (offline):
  tar xzf ~/$(basename "${BUNDLE}")
  ORT_SOURCE_DIR=~/offline-bundle/onnxruntime \\
      src/yolov8_ros/yolo_onnxruntime_vendor/scripts/build_ort160_aarch64.sh
  colcon build --symlink-install --packages-select yolo_msgs yolo_onnxruntime_vendor yolo_ros
  source install/setup.bash
  ros2 launch yolo_bringup yolo.launch.py model:=~/offline-bundle/models/yolo26m.onnx
EOF
