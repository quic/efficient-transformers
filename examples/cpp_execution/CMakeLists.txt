# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

project(InferenceSetIOBuffer)
cmake_minimum_required (VERSION 3.15)

# Find the pybind11 CMake directory using a Python script
execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set the CMAKE_PREFIX_PATH to include pybind11
set(CMAKE_PREFIX_PATH ${pybind11_DIR} ${CMAKE_PREFIX_PATH})
set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 REQUIRED)

pybind11_add_module(InferenceSetIOBuffer MODULE InferenceSetIOBuffer.cpp)


include_directories("/opt/qti-aic/dev/inc")
include_directories("examples/cpp_execution")

target_link_libraries(InferenceSetIOBuffer PRIVATE ${PYTHON_LIBRARIES} pybind11::module pthread dl)

target_include_directories(InferenceSetIOBuffer PRIVATE ${PYTHON_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

set_target_properties(
    InferenceSetIOBuffer
    PROPERTIES
    LINK_FLAGS "-Wl,--no-as-needed"
)

set(CMAKE_BUILD_TYPE Debug)

target_compile_options(InferenceSetIOBuffer PRIVATE
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
message(STATUS "pybind11_DIR: ${pybind11_DIR}")
