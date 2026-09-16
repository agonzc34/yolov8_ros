# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.8)
project(yolo_hfhub_vendor)

# huggingface-hub-cpp requires C++17 (std::variant, std::filesystem). Do not rely
# on the compiler default: GCC 9 (Ubuntu 20.04 / Galactic) defaults to gnu++14.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
find_package(ament_cmake REQUIRED)
find_package(CURL REQUIRED)

# Fetch the huggingface-hub-cpp library (an unofficial C++ client for the
# Hugging Face Hub), used at runtime to download YOLO models from the Hub.
FetchContent_Declare(
  yolo_hfhub
  GIT_REPOSITORY https://github.com/agonzc34/huggingface-hub-cpp
  GIT_TAG        1.1.4
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(yolo_hfhub)

# Enforce C++17 on the fetched target even if it sets its own standard.
set_target_properties(hfhub PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Export the hfhub target and its include directories for downstream packages.
install(
  TARGETS hfhub
  EXPORT export_yolo_hfhub
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

ament_export_include_directories(include)
ament_export_dependencies(CURL)
ament_export_targets(export_yolo_hfhub HAS_LIBRARY_TARGET)

install(FILES
  LICENSE
  THIRD_PARTY_NOTICES.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
