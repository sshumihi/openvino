# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_xpu_plugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "XPU"
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR plugin.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::runtime::dev
    openvino::zero_loader
    ze_loader)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})

ov_register_plugins(MAIN_TARGET ${TARGET_NAME})
