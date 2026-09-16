#!/usr/bin/env bash
# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT
#
# Build ONNX Runtime 1.6.0 with CUDA + TensorRT for the aarch64 Jetson robot and
# package it in the flat layout consumed by yolo_onnxruntime_vendor.
#
# Run this on the robot. It can work fully offline when given a source tree
# prepared by scripts/prepare_offline_bundle.sh (see that script for the
# host-side workflow).
#
# Usage: scripts/build_ort160_aarch64.sh [output_dir]
# Default output_dir: <package>/ort160
#
# Source acquisition (first match wins):
#   ORT_SOURCE_DIR=<path>     an existing onnxruntime source tree (contains build.sh)
#   ORT_SOURCE_TARBALL=<path> a tarball whose top level contains onnxruntime/
#   ${ORT_BUILD_DIR}/onnxruntime   a previously extracted/cloned tree
#   otherwise                 git clone --recursive (requires internet)
#
# Env overrides: ORT_CUDA_ARCH, CUDA_HOME, CUDNN_HOME, TENSORRT_HOME,
#                ORT_BUILD_DIR
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

for tool in cmake python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "error: ${tool} not found" >&2
    exit 1
  }
done
[[ -d "${CUDA_HOME}" ]] || {
  echo "error: CUDA not found at ${CUDA_HOME}" >&2
  exit 1
}

# --- CUDA host-compiler compatibility -----------------------------------------
# nvcc only supports host compilers up to a version that depends on the CUDA
# release (CUDA 10.2 -> gcc <= 8). Distros with a newer default gcc (9+ on
# Ubuntu 20.04) make nvcc fail with "unsupported GNU version". Auto-select an
# older g++ when needed; override with CUDAHOSTCXX to skip the detection.
NVCC="${CUDA_HOME}/bin/nvcc"
[[ -x "${NVCC}" ]] || {
  echo "error: nvcc not found at ${NVCC}" >&2
  exit 1
}
CUDA_MAJOR="$("${NVCC}" --version | sed -n 's/.*release \([0-9]\+\).*/\1/p')"
case "${CUDA_MAJOR}" in
  9) CUDA_MAX_GCC=7 ;;
  10) CUDA_MAX_GCC=8 ;;
  11) CUDA_MAX_GCC=10 ;;
  *) CUDA_MAX_GCC=99 ;;
esac
DEFAULT_GCC_MAJOR="$(g++ -dumpversion | cut -d. -f1)"
if [[ -z "${CUDAHOSTCXX:-}" && "${DEFAULT_GCC_MAJOR}" -gt "${CUDA_MAX_GCC}" ]]; then
  host_cxx=""
  host_ver=""
  for ver in "${CUDA_MAX_GCC}" "$((CUDA_MAX_GCC - 1))" 7 6; do
    if command -v "g++-${ver}" >/dev/null 2>&1; then
      host_cxx="$(command -v "g++-${ver}")"
      host_ver="${ver}"
      break
    fi
  done
  if [[ -z "${host_cxx}" ]]; then
    echo "error: CUDA ${CUDA_MAJOR} supports host gcc <= ${CUDA_MAX_GCC}, but the" >&2
    echo "       default is gcc ${DEFAULT_GCC_MAJOR}. Install a compatible one, e.g.:" >&2
    echo "         sudo apt install gcc-${CUDA_MAX_GCC} g++-${CUDA_MAX_GCC}" >&2
    echo "       or export CUDAHOSTCXX=/usr/bin/g++-${CUDA_MAX_GCC}" >&2
    exit 1
  fi
  export CXX="${host_cxx}"
  export CUDAHOSTCXX="${host_cxx}"
  if host_cc="$(command -v "gcc-${host_ver}" 2>/dev/null)"; then
    export CC="${host_cc}"
  fi
  echo "==> Default g++ ${DEFAULT_GCC_MAJOR} is too new for CUDA ${CUDA_MAJOR};" \
    "using g++-${host_ver} (${host_cxx})"
fi
CUDAHOSTCXX="${CUDAHOSTCXX:-$(command -v g++)}"
echo "==> CUDA host compiler: ${CUDAHOSTCXX} (CC=${CC:-$(command -v gcc)}, CXX=${CXX:-$(command -v g++)})"

mkdir -p "${WORK_DIR}"
SRC_DIR="${WORK_DIR}/onnxruntime"

if [[ -n "${ORT_SOURCE_TARBALL:-}" ]]; then
  echo "==> Extracting source from ${ORT_SOURCE_TARBALL}"
  rm -rf "${SRC_DIR}"
  tar -xf "${ORT_SOURCE_TARBALL}" -C "${WORK_DIR}"
elif [[ -n "${ORT_SOURCE_DIR:-}" ]]; then
  SRC_DIR="${ORT_SOURCE_DIR}"
elif [[ -d "${SRC_DIR}" ]]; then
  echo "==> Reusing existing source tree at ${SRC_DIR}"
else
  command -v git >/dev/null 2>&1 || {
    echo "error: no local ONNX Runtime source and git is unavailable." >&2
    echo "       Run prepare_offline_bundle.sh on an internet host first." >&2
    exit 1
  }
  echo "==> Cloning ONNX Runtime v${ORT_VERSION} (requires internet)"
  # --recursive is mandatory: the GitHub source tarball has no submodules.
  git clone --recursive -b "v${ORT_VERSION}" \
    https://github.com/microsoft/onnxruntime "${SRC_DIR}"
fi

[[ -f "${SRC_DIR}/build.sh" ]] || {
  echo "error: no build.sh under '${SRC_DIR}'. Expected onnxruntime/ at the" >&2
  echo "       source root (e.g. ORT_SOURCE_DIR=<bundle>/onnxruntime)." >&2
  exit 1
}

echo "==> Building ONNX Runtime ${ORT_VERSION} (CUDA ${CUDA_HOME}, TRT ${TRT_HOME})"
if [[ -d "${SRC_DIR}/build" ]]; then
  echo "    note: ${SRC_DIR}/build already exists; if a previous attempt used a" \
    "different compiler, remove it first (rm -rf ${SRC_DIR}/build)."
fi
pushd "${SRC_DIR}"
# --skip_submodule_sync keeps the build offline (prepare_offline_bundle.sh has
# already populated every submodule).
./build.sh --config Release --update --build --build_shared_lib --skip_tests \
  --skip_submodule_sync --parallel \
  --use_tensorrt --tensorrt_home "${TRT_HOME}" \
  --use_cuda --cuda_home "${CUDA_HOME}" --cudnn_home "${CUDNN_HOME}" \
  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}" \
  --cmake_extra_defines "CMAKE_CUDA_HOST_COMPILER=${CUDAHOSTCXX}" \
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
