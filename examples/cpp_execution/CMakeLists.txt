# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

project(InferenceSetIOBufferExample)
cmake_minimum_required (VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

pybind11_add_module(InferenceSetIOBufferExample MODULE InferenceSetIOBufferExample.cpp)


include_directories("/opt/qti-aic/dev/inc")
include_directories("examples/cpp_execution")

target_link_libraries(InferenceSetIOBufferExample PRIVATE ${PYTHON_LIBRARIES} pybind11::module pthread dl)

target_include_directories(InferenceSetIOBufferExample PRIVATE ${PYTHON_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

set_target_properties(
    InferenceSetIOBufferExample
    PROPERTIES
    LINK_FLAGS "-Wl,--no-as-needed"
    CFLAGS "-g -Wall -Wextra"
)

set(CMAKE_BUILD_TYPE Debug)

target_compile_options(InferenceSetIOBufferExample PRIVATE
                    -fstack-protector-all
                    -fstack-protector-all
                    -Werror
                    -Wall
                    -Wextra
                    -Wunused-variable
                    -Wunused-parameter
                    -Wnon-virtual-dtor
                    -Wno-missing-field-initializers)

#Print paths for debugging
message(STATUS "Python Include Dirs: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Python Include Dirs: ${PYTHON_LIBRARIES}")
