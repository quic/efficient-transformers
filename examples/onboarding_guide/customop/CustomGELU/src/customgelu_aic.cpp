//-----------------------------------------------------------------------------
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//-----------------------------------------------------------------------------

#include "CustomOpAICInterface.h"
#include "stddef.h"

extern "C" {

/* The AIC compilation target supports an API similar to the Interpreter API.
Additionally, threadId, which is the AIC thread ID, is passed.
Kernel is invoked by four AIC threads with threadId equal to 0, 1, 2, and 3. */

void CustomGELUAIC(
	const CustomOpContext *ctx, 
	const int32_t threadId)
{
}

}
