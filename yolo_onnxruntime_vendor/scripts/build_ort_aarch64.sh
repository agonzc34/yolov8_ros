#!/usr/bin/env bash
# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT
#
# Build ONNX Runtime (any version) with CUDA + TensorRT for the aarch64 Jetson
# robot and package it in the flat layout consumed by yolo_onnxruntime_vendor
# (point that package at the result with -DONNXRUNTIME_ROOT=<output_dir>).
#
# Run this on the robot. It can work fully offline when given a source tree
# prepared by scripts/prepare_offline_bundle.sh (see that script for the
# host-side workflow).
#
# Usage: scripts/build_ort_aarch64.sh <ort_version> [output_dir]
#   ort_version   ONNX Runtime release to build, e.g. 1.6.0 or 1.20.2
#                 (the git tag is v<ort_version>).
#   output_dir    defaults to <package>/ort-<ort_version>
#
# Required environment:
#   ORT_CUDA_ARCH   target CUDA architecture. Jetson Xavier = 72, Orin = 87.
#
# Source acquisition (first match wins):
#   ORT_SOURCE_DIR=<path>     an existing onnxruntime source tree (contains build.sh)
#   ORT_SOURCE_TARBALL=<path> a tarball whose top level contains onnxruntime/
#   ${ORT_BUILD_DIR}/onnxruntime   a previously extracted/cloned tree
#   otherwise                 git clone --recursive (requires internet)
#
# Env overrides: CUDA_HOME, CUDNN_HOME, TENSORRT_HOME, ORT_BUILD_DIR
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") <ort_version> [output_dir]" >&2
}

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "error: ONNX Runtime version is required" >&2
  usage
  exit 2
fi

ORT_VERSION="${1#v}"
if [[ ! "${ORT_VERSION}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
  echo "error: '${ORT_VERSION}' does not look like a version (e.g. 1.20.2)" >&2
  usage
  exit 2
fi

CUDA_ARCH="${ORT_CUDA_ARCH:-}"
if [[ -z "${CUDA_ARCH}" ]]; then
  echo "error: ORT_CUDA_ARCH must be set (e.g. 72 for Xavier, 87 for Orin)" >&2
  exit 2
fi

# JetPack commonly has no /usr/local/cuda symlink; fall back to the newest
# /usr/local/cuda-<version> when CUDA_HOME is not given.
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -d /usr/local/cuda ]]; then
    CUDA_HOME=/usr/local/cuda
  else
    CUDA_HOME="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)"
  fi
fi
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDNN_HOME="${CUDNN_HOME:-/usr/lib/aarch64-linux-gnu}"
TRT_HOME="${TENSORRT_HOME:-/usr/lib/aarch64-linux-gnu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${2:-${SCRIPT_DIR}/../ort-${ORT_VERSION}}"
WORK_DIR="${ORT_BUILD_DIR:-${SCRIPT_DIR}/../ort-${ORT_VERSION}-build}"

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

# ONNX Runtime 1.16+ needs CMake >= 3.26 (JetPack/Ubuntu 22.04 ship 3.22, which
# is too old). Older ONNX Runtime builds accept much older CMake.
version_ge() { [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" == "$1" ]]; }
cuda_min_cmake=3.13
if version_ge "${ORT_VERSION}" 1.16; then
  cuda_min_cmake=3.26
fi
MIN_CMAKE="${ORT_MIN_CMAKE:-${cuda_min_cmake}}"
CMAKE_VER="$(cmake --version | sed -n 's/^cmake version //p')"
if ! version_ge "${CMAKE_VER}" "${MIN_CMAKE}"; then
  echo "error: ONNX Runtime ${ORT_VERSION} needs CMake >= ${MIN_CMAKE}, found ${CMAKE_VER}" >&2
  echo "       If you used the offline bundle, add its CMake to PATH first:" >&2
  echo "         export PATH=~/offline-bundle-*/tools/cmake-*-linux-aarch64/bin:\$PATH" >&2
  echo "       otherwise install a newer CMake (e.g. https://apt.kitware.com/)," >&2
  echo "       or set ORT_MIN_CMAKE to override this check." >&2
  exit 1
fi
echo "==> CMake ${CMAKE_VER} (>= ${MIN_CMAKE} required)"

[[ -d "${CUDA_HOME}" ]] || {
  echo "error: CUDA toolkit not found (CUDA_HOME='${CUDA_HOME}')" >&2
  echo "       found: $(ls -d /usr/local/cuda* 2>/dev/null | tr '\n' ' ' || true)" >&2
  echo "       set CUDA_HOME to the toolkit root, e.g. /usr/local/cuda-12.6" >&2
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

echo "==> Building ONNX Runtime ${ORT_VERSION} (CUDA ${CUDA_HOME}, TRT ${TRT_HOME}," \
  "arch ${CUDA_ARCH})"
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

# Copy every produced library. The provider set differs across versions: 1.6
# links CUDA into the TensorRT provider, while >= 1.12 ships a separate
# libonnxruntime_providers_cuda.so. Globbing keeps this version-agnostic.
shopt -s nullglob
built_libs=("${BUILD_DIR}"/libonnxruntime*.so*)
shopt -u nullglob
if [[ ${#built_libs[@]} -eq 0 ]]; then
  echo "error: no libonnxruntime*.so* found in ${BUILD_DIR}" >&2
  exit 1
fi
cp -av "${built_libs[@]}" "${OUT_DIR}/lib/"

# Header names are stable across releases but their location under include/ is
# not, so look each one up by basename (flat layout expected by the vendor).
copy_header() {
  local name="$1" required="$2" src
  src="$(find "${SRC_DIR}/include" -name "${name}" -print -quit)"
  if [[ -n "${src}" ]]; then
    cp "${src}" "${OUT_DIR}/include/"
  elif [[ "${required}" == "required" ]]; then
    echo "error: required header ${name} not found under ${SRC_DIR}/include" >&2
    exit 1
  else
    echo "warning: header ${name} not found; skipping" >&2
  fi
}

copy_header onnxruntime_c_api.h required
copy_header onnxruntime_cxx_api.h required
copy_header onnxruntime_cxx_inline.h required
copy_header onnxruntime_float16.h optional
copy_header onnxruntime_session_options_config_keys.h optional
copy_header onnxruntime_run_options_config_keys.h optional
copy_header provider_options.h optional
copy_header cpu_provider_factory.h optional
copy_header cuda_provider_factory.h optional
copy_header tensorrt_provider_factory.h optional

for notice in LICENSE ThirdPartyNotices.txt; do
  if [[ -f "${SRC_DIR}/${notice}" ]]; then
    cp "${SRC_DIR}/${notice}" "${OUT_DIR}/${notice}"
  else
    echo "warning: ${notice} not found in the source tree; skipping" >&2
  fi
done

echo "ONNX Runtime ${ORT_VERSION} installed to ${OUT_DIR}"
echo "Consume it with: colcon build --cmake-args -DONNXRUNTIME_ROOT=${OUT_DIR}"
