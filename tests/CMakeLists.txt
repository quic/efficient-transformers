# -----------------------------------------------------------------------------
#
#Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

add_test(UnitTestSuite
      ${Python3_EXECUTABLE} -m pytest -s ${CMAKE_CURRENT_SOURCE_DIR} --junitxml=${CMAKE_CURRENT_BINARY_DIR}/py_unit_test_output.xml
      )

set_property(TEST UnitTestSuite
      APPEND PROPERTY ENVIRONMENT "${LLM_PYTHONPATH}")
