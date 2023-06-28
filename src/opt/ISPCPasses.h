/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file ISPCPasses.h
    @brief Includes available ISPC passes
*/

#pragma once

#include "CheckIRForXeTarget.h"
#include "GatherCoalescePass.h"
#include "ImproveMemoryOps.h"
#include "InstructionSimplify.h"
#include "IntrinsicsOptPass.h"
#include "IsCompileTimeConstant.h"
#include "MakeInternalFuncsStatic.h"
#include "MangleOpenCLBuiltins.h"
#include "PeepholePass.h"
#include "ReplacePseudoMemoryOps.h"
#include "ReplaceStdlibShiftPass.h"
#include "XeGatherCoalescePass.h"
#include "XeReplaceLLVMIntrinsics.h"
