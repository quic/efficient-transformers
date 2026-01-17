//-----------------------------------------------------------------------------
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//-----------------------------------------------------------------------------

#include "CustomOpFunctions.h"
#include "CustomOpInterpreterInterface.h"
#include "CustomOpTileConfigHelpers.h"
#include "CustomOpTypes.h"

#include <string.h>

extern "C" {
bool customOpVerify(
	const CustomOpPropertiesHandle *const opProp)
{
	/* Refer to function declaration at CustomOpFunctions.h for usage. */
	
	return true;
}

const char * customOpSelectImpl(
	const CustomOpPropertiesHandle *const opProp, 
	const CustomOpKernelInfo *const kernelInfos, 
	const int32_t numKernels, 
	const char *backend)
{
	/* Refer to function declaration at CustomOpFunctions.h for usage. */
	
	/* For AIC pick '<OpName>AIC', for Interpreter pick '<OpName>Interpreter' */
	if (strcmp(backend, "AIC") == 0)
	{
		return "";
	}
	else if (strcmp(backend, "Interpreter") == 0)
	{
		return "";
	}
	return nullptr;
}

bool customOpInferShape(
	CustomOpPropertiesHandle *const opProp)
{
	/* Refer to function declaration at CustomOpFunctions.h for usage. */
	
	return false;
}

bool customOpSetProperties(
	CustomOpPropertiesHandle *opProp)
{
	/* Refer to function declaration at CustomOpFunctions.h for usage. */
	
	return false;
}

bool customOpMapTiles(
	CustomOpPropertiesHandle *opProp)
{
	/* Refer to function declaration at CustomOpFunctions.h for usage. */
	
	return false;
}
void customOpDeallocateMemory(
	CustomOpPropertiesHandle *opProp)
{
	/* Refer to function declaration at CustomOpFunctions.h for usage. */
	
	CustomOpTileConfigHelpers::destroyTileConfigsAndMergeConfigs(opProp);
}
}
