//-----------------------------------------------------------------------------
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
//-----------------------------------------------------------------------------

/*
* This file can be compiled separately and can be loaded using dlopen
* Compilation command: (tried with gcc 5.5)
* g++ -shared -std=c++11 -fPIC -o <opName>_lib.so <opName_functions>.cpp <opName_interpreter>.cpp -I/opt/qti-aic/dev/inc
* for example: g++ -shared -std=c++11 -fPIC -o reluop_lib.so reluop_functions.cpp reluop_interpreter.cpp -I/opt/qti-aic/dev/inc
*/

#include "CustomOpInterpreterInterface.h"

extern "C" {
void CustomGELUInterpreter(
	CustomOpContext *ctx)
{
	/* The interpreter implementation is provided to the compiler as a shared library
	(or collection of shared libraries). Each shared library can contain multiple
	versions (flavors) of implementations of the operation, refered onwards as kernels.
	A kernel is selected at model compilation time by the selection function. The
	developer is responsible for compilation of these shared libraries. As the interface
	is C, the shared libraries can be compiled by various compilers (GCC, CLANG, etc).
	In addition, as these shared libraries are running on the Host CPU, the developer
	can open files, dump results, use stdout/stderr for printing debug messages, etc.
	This makes the Interpreter implementation a very effective way for debugging the
	operation functionality as part of model execution. The signature of the
	kernel (implementation) is generic, and fits any custom operation. */
}
}
