/*
  Copyright (c) 2010-2016, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @file main.cpp
    @brief main() entrypoint implementation for ispc
*/

#include "ispc.h"
#include "module.h"
#include "util.h"
#include "type.h"
#include "options.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef ISPC_IS_WINDOWS
  #include <time.h>
#else
  #include <unistd.h>
#endif // ISPC_IS_WINDOWS
#include <llvm/Support/Signals.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>


static void
lSignal(void *) {
    FATAL("Unhandled signal sent to process; terminating.");
}

int main(int Argc, char *Argv[]) {
    llvm::sys::AddSignalHandler(lSignal, NULL);

    // initialize available LLVM targets
#ifndef __arm__
    // FIXME: LLVM build on ARM doesn't build the x86 targets by default.
    // It's not clear that anyone's going to want to generate x86 from an
    // ARM host, though...
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();
    LLVMInitializeX86TargetMC();
#endif // !__ARM__

#ifdef ISPC_ARM_ENABLED
    // Generating ARM from x86 is more likely to be useful, though.
    LLVMInitializeARMTargetInfo();
    LLVMInitializeARMTarget();
    LLVMInitializeARMAsmPrinter();
    LLVMInitializeARMAsmParser();
    LLVMInitializeARMDisassembler();
    LLVMInitializeARMTargetMC();
#endif

#ifdef ISPC_NVPTX_ENABLED
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXAsmPrinter();
    LLVMInitializeNVPTXTargetMC();
#endif /* ISPC_NVPTX_ENABLED */

    OptionParseResult opr = ParseOptions(Argc, Argv);
    int ret = (opr & OPTION_PARSE_RESULT_ERROR) ? 1 : 0;
    switch (opr) {
    case OPTION_PARSE_RESULT_REQUEST_VERSION:
        PrintVersion();
        exit(0);
    case OPTION_PARSE_RESULT_REQUEST_USAGE:
        PrintUsage(ret);
    case OPTION_PARSE_RESULT_REQUEST_DEV_USAGE:
        PrintDevUsage(ret);
    default:
        break;
    }

    if (g->outFileName == NULL &&
        g->headerFileName == NULL &&
        g->depsFileName == NULL &&
        g->hostStubFileName == NULL &&
        g->devStubFileName == NULL)
      Warning(SourcePos(), "No output file or header file name specified. "
              "Program will be compiled and warnings/errors will "
              "be issued, but no output will be generated.");

    return Module::CompileAndOutput(g->fileName,
                                    g->archName, g->cpuName, g->targetName,
                                    g->generatePIC,
                                    g->outputType,
                                    g->outFileName,
                                    g->headerFileName,
                                    g->includeFileName,
                                    g->depsFileName,
                                    g->hostStubFileName,
                                    g->devStubFileName);
}
