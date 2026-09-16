cmake_minimum_required(VERSION 3.8)
project(yolo_onnxruntime_vendor)

# This branch targets the aarch64 Jetson robot only. ONNX Runtime 1.6.0 is built
# from source by scripts/build_ort160_aarch64.sh (there is no official aarch64
# binary) and consumed from ONNXRUNTIME_ROOT or the default local prefix.
if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  message(FATAL_ERROR
    "yolo_onnxruntime_vendor on this branch supports aarch64 only "
    "(got '${CMAKE_SYSTEM_PROCESSOR}'). Use the yolo_onnx_cpp branch for x86_64.")
endif()

find_package(ament_cmake REQUIRED)

set(ONNXRUNTIME_DEFAULT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ort160")
if(ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_INSTALL_DIR "${ONNXRUNTIME_ROOT}")
elseif(EXISTS "${ONNXRUNTIME_DEFAULT_ROOT}/lib/libonnxruntime.so")
  set(ONNXRUNTIME_INSTALL_DIR "${ONNXRUNTIME_DEFAULT_ROOT}")
else()
  message(FATAL_ERROR
    "ONNX Runtime 1.6 not found. Run "
    "'${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_ort160_aarch64.sh' first, or "
    "configure with -DONNXRUNTIME_ROOT=/path/to/onnxruntime.")
endif()

if(NOT EXISTS "${ONNXRUNTIME_INSTALL_DIR}/lib/libonnxruntime.so")
  message(FATAL_ERROR
    "ONNXRUNTIME_ROOT='${ONNXRUNTIME_INSTALL_DIR}' has no lib/libonnxruntime.so")
endif()

install(DIRECTORY ${ONNXRUNTIME_INSTALL_DIR}/lib DESTINATION .)
install(DIRECTORY ${ONNXRUNTIME_INSTALL_DIR}/include DESTINATION .)

install(FILES
  LICENSE
  THIRD_PARTY_NOTICES.md
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  ${ONNXRUNTIME_INSTALL_DIR}/LICENSE
  ${ONNXRUNTIME_INSTALL_DIR}/ThirdPartyNotices.txt
  DESTINATION share/${PROJECT_NAME}/onnxruntime
)

ament_export_include_directories(include)
ament_export_libraries(onnxruntime)

ament_package()
