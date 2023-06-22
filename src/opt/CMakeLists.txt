#
#  Copyright (c) 2022-2023, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

target_sources(${PROJECT_NAME}
    PRIVATE
        "CheckIRForXeTarget.cpp"
        "CheckIRForXeTarget.h"
        "GatherCoalescePass.cpp"
        "GatherCoalescePass.h"
        "IsCompileTimeConstant.cpp"
        "IsCompileTimeConstant.h"
        "ImproveMemoryOps.cpp"
        "ImproveMemoryOps.h"
        "InstructionSimplify.cpp"
        "InstructionSimplify.h"
        "IntrinsicsOptPass.cpp"
        "IntrinsicsOptPass.h"
        "MangleOpenCLBuiltins.cpp"
        "MangleOpenCLBuiltins.h"
        "MakeInternalFuncsStatic.cpp"
        "MakeInternalFuncsStatic.h"
        "PeepholePass.cpp"
        "PeepholePass.h"
        "ReplacePseudoMemoryOps.cpp"
        "ReplacePseudoMemoryOps.h"
        "ReplaceStdlibShiftPass.cpp"
        "ReplaceStdlibShiftPass.h"
        "XeGatherCoalescePass.cpp"
        "XeGatherCoalescePass.h"
        "XeReplaceLLVMIntrinsics.cpp"
        "XeReplaceLLVMIntrinsics.h"
)
