#!/usr/bin/env bash
# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT
#
# Build ONNX Runtime 1.6.0 with CUDA + TensorRT for the aarch64 Jetson robot and
# package it in the flat layout consumed by yolo_onnxruntime_vendor.
#
# Usage: scripts/build_ort160_aarch64.sh [output_dir]
# Default output_dir: <package>/ort160
set -euo pipefail

ORT_VERSION="1.6.0"
CUDA_ARCH="${ORT_CUDA_ARCH:-72}" # Jetson Xavier = 7.2
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDNN_HOME="${CUDNN_HOME:-/usr/lib/aarch64-linux-gnu}"
TRT_HOME="${TENSORRT_HOME:-/usr/lib/aarch64-linux-gnu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/../ort160}"
WORK_DIR="${ORT_BUILD_DIR:-${SCRIPT_DIR}/../ort160-build}"

ARCH="$(uname -m)"
if [[ "${ARCH}" != "aarch64" ]]; then
  echo "error: this script supports aarch64 only (got ${ARCH})" >&2
  exit 1
fi

for tool in git cmake python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "error: ${tool} not found" >&2
    exit 1
  }
done
[[ -d "${CUDA_HOME}" ]] || {
  echo "error: CUDA not found at ${CUDA_HOME}" >&2
  exit 1
}

mkdir -p "${WORK_DIR}"
SRC_DIR="${WORK_DIR}/onnxruntime"
# --recursive is mandatory: the GitHub source tarball has no submodules.
if [[ ! -d "${SRC_DIR}/.git" ]]; then
  git clone --recursive -b "v${ORT_VERSION}" \
    https://github.com/microsoft/onnxruntime "${SRC_DIR}"
fi

pushd "${SRC_DIR}"
./build.sh --config Release --update --build --build_shared_lib --skip_tests --parallel \
  --use_tensorrt --tensorrt_home "${TRT_HOME}" \
  --use_cuda --cuda_home "${CUDA_HOME}" --cudnn_home "${CUDNN_HOME}" \
  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF
popd

BUILD_DIR="${SRC_DIR}/build/Linux/Release"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}/lib" "${OUT_DIR}/include"

cp -av "${BUILD_DIR}"/libonnxruntime.so* "${OUT_DIR}/lib/"
cp -av "${BUILD_DIR}"/libonnxruntime_providers_tensorrt.so* "${OUT_DIR}/lib/"
cp -av "${BUILD_DIR}"/libonnxruntime_providers_shared.so* "${OUT_DIR}/lib/"

cp "${SRC_DIR}/include/onnxruntime/core/session/onnxruntime_c_api.h" "${OUT_DIR}/include/"
cp "${SRC_DIR}/include/onnxruntime/core/session/onnxruntime_cxx_api.h" "${OUT_DIR}/include/"
cp "${SRC_DIR}/include/onnxruntime/core/session/onnxruntime_cxx_inline.h" "${OUT_DIR}/include/"
cp "${SRC_DIR}/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h" "${OUT_DIR}/include/"
cp "${SRC_DIR}/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h" "${OUT_DIR}/include/"
cp "${SRC_DIR}/include/onnxruntime/core/providers/cuda/cuda_provider_factory.h" "${OUT_DIR}/include/"
cp "${SRC_DIR}/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h" "${OUT_DIR}/include/"

cp "${SRC_DIR}/LICENSE" "${OUT_DIR}/LICENSE"
cp "${SRC_DIR}/ThirdPartyNotices.txt" "${OUT_DIR}/ThirdPartyNotices.txt"

echo "ONNX Runtime ${ORT_VERSION} installed to ${OUT_DIR}"
