# -----------------------------------------------------------------------------
#
#Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

# Set the minimum version of CMake required to build the project
cmake_minimum_required(VERSION 3.0)

# Set the name of the project
project(custom_lib)

# Set the C++ standard to use
set(CMAKE_CXX_STANDARD 11)

# Set the source files for the library
set(SOURCE_FILES ${SRC_FILE})

# Set the include directories for the library
include_directories(/opt/qti-aic/dev/inc)

# Create the shared library
add_library(${CUSTOM_LIB} SHARED ${SOURCE_FILES})

