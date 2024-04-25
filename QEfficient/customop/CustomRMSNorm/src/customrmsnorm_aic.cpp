/* -----------------------------------------------------------------------------
 *
 * Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * -----------------------------------------------------------------------------
 */

/* Interface version: 5.0.0 */
#include "CustomOpAICInterface.h"
#include "stddef.h"
#include "CustomOpLog.h"
#include <cmath>
#include <hexagon_protos.h>
#include <hexagon_types.h>
// #include "defs.h"
#define NUM_THREADS (4)
extern "C" {
/* The AIC compilation target supports an API similar to the Interpreter
API. Additionally, threadId, which is the AIC thread ID, is passed.
Kernel is invoked by four AIC threads with threadId equal to 0, 1, 2, and 3. */
void CustomRMSNormAIC(const CustomOpContext *context, const int32_t threadId) {
    int32_t *input_dims = context->inputs[0].sizes;
    int32_t input_rank = context->inputs[0].rank;
    int32_t batch_size = input_dims[0];
    int32_t sequence_length = input_dims[1];
    int32_t hidden_size = input_dims[2];
    float eps = *(float *)context->params[0].data;
    float16_ty *hidden_states = (float16_ty *)context->inputs[0].data;
    float16_ty *weight = (float16_ty *)context->inputs[1].data;
    float16_ty *output = (float16_ty *)context->outputs[0].data;
    // Calculate reciprocal hidden size to avoid division multiple times
    float r_hidden_size = 1.0f / hidden_size;
    for (int i = threadId; i < sequence_length; i += NUM_THREADS) {
        float variance = 0.0f;
        int layer_offset = i * hidden_size;
        for (int j = 0; j < hidden_size; j++) {
            float val = hidden_states[j + layer_offset];
            variance += val * val;
        }
        variance *= r_hidden_size;
        float rms = sqrt(variance + eps);
        float r_rms = 1.0f / rms;
        for (int j = 0; j < hidden_size; j++){
            output[j + layer_offset] = hidden_states[j + layer_offset] * r_rms * weight[j];
        }
    }
  }
}
