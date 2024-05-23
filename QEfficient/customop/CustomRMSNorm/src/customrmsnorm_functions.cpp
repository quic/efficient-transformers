/* -----------------------------------------------------------------------------
 *
 * Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * -----------------------------------------------------------------------------
 */

/* Interface version: 5.0.0 */
#include "CustomOpFunctions.h"
#include "CustomOpInterpreterInterface.h"
#include "CustomOpTypes.h"
#include <string.h>
extern "C" {
bool customOpVerify(const CustomOpProperties *const opProp) {
  /* Refer to function declaration at CustomOpFunctions.h for usage. */
  // Must have two params.
  if (opProp->params.size() < 1)
    return false;
  auto &param0 = opProp->params[0];
  // Check params names are valid.
  if (strcmp(param0.name, "epsilon"))
    return false;
  // Op must have only 2 input and 1 output.
  if (opProp->inputs.size() != 2 || opProp->outputs.size() != 1)
    return false;
  // Input and Output must have the same data type.
  if (opProp->inputs[0].dtype != opProp->outputs[0].dtype)
    return false;
  // Input and Output must have the same dimensions.
  if (opProp->inputs[0].rank != opProp->outputs[0].rank)
    return false;
  for (int i = 0; i < opProp->inputs[0].rank; i++) {
    if (opProp->inputs[0].sizes[i] != opProp->outputs[0].sizes[i])
      return false;
  }
  return true;
}
const char *customOpSelectImpl(const CustomOpProperties *const opProp,
                               const CustomOpKernelInfo *const kernelInfos,
                               const int32_t numKernels, const char *backend) {
  /* Refer to function declaration at CustomOpFunctions.h for usage. */
  /* For AIC pick '<OpName>AIC', for Interpreter pick '<OpName>Interpreter' */
  if (strcmp(backend, "AIC") == 0) {
    return "CustomRMSNormAIC";
  } else if (strcmp(backend, "Interpreter") == 0) {
    return "CustomReluInterpreter";
  }
}
bool customOpInferShape(CustomOpProperties *const opProp) {
  /* Refer to function declaration at CustomOpFunctions.h for usage. */
  if (opProp->inputs.size() != 2 || opProp->outputs.size() != 1)
    return false;
  // There is only 1 output.
  // Output has the same type as input.
  CustomOpIOTensor &out = opProp->outputs[0];
  out.rank = opProp->inputs[0].rank;
  for (int i = 0; i < opProp->inputs[0].rank; i++) {
    out.sizes[i] = opProp->inputs[0].sizes[i];
  }
  out.dtype = opProp->inputs[0].dtype;
  return true;
}
bool customOpSetProperties(CustomOpProperties *const opProp) {
  /* Refer to function declaration at CustomOpFunctions.h for usage. */
  if (opProp->inputs[0].sizes[0] > 1)
  {
    setTileConfig(opProp, "output", {0, 1});
    return true;
  }
  return false;
}
bool customOpMapTiles(CustomOpProperties *const opProp) {
  /* Refer to function declaration at CustomOpFunctions.h for usage. */
   // Get output start and end indices
  if (opProp->inputs[0].sizes[0] > 1)
  {
    const std::vector<int32_t> startIndices = tileStartIndices(opProp->outputs[0]);
    const std::vector<int32_t> endIndices = tileEndIndices(opProp->outputs[0]);
    createInputTile(opProp, 0, startIndices, endIndices);
    return true;
  }
  return false;
}
}
