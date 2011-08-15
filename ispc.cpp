/*
  Copyright (c) 2010-2011, Intel Corporation
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

/** @file ispc.cpp
    @brief ispc global definitions
*/

#include "ispc.h"
#include "module.h"
#include "util.h"
#include <stdio.h>
#ifdef ISPC_IS_WINDOWS
#include <windows.h>
#include <direct.h>
#endif
#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#ifndef LLVM_2_8
#include <llvm/Analysis/DIBuilder.h>
#endif
#include <llvm/Analysis/DebugInfo.h>
#include <llvm/Support/Dwarf.h>

Globals *g;
Module *m;

///////////////////////////////////////////////////////////////////////////
// Target

Target::Target() {
    arch = "x86-64";
    cpu = "nehalem";
    is32bit = false;
    isa = SSE4;
    nativeVectorWidth = 4;
    vectorWidth = 4;
}

///////////////////////////////////////////////////////////////////////////
// Opt

Opt::Opt() {
    level = 1;
    fastMath = false;
    disableBlendedMaskedStores = false;
    disableCoherentControlFlow = false;
    disableUniformControlFlow = false;
    disableGatherScatterOptimizations = false;
    disableMaskedStoreToStore = false;
    disableGatherScatterFlattening = false;
    disableUniformMemoryOptimizations = false;
    disableMaskedStoreOptimizations = false;
}

///////////////////////////////////////////////////////////////////////////
// Globals

Globals::Globals() {
    mathLib = Globals::Math_ISPC;

    includeStdlib = true;
    runCPP = true;
    debugPrint = false;
    disableWarnings = false;
    emitPerfWarnings = true;
    emitInstrumentation = false;
    generateDebuggingSymbols = false;

    ctx = new llvm::LLVMContext;

#ifdef ISPC_IS_WINDOWS
    _getcwd(currentDirectory, sizeof(currentDirectory));
#else
    getcwd(currentDirectory, sizeof(currentDirectory));
#endif
}

///////////////////////////////////////////////////////////////////////////
// ASTNode

ASTNode::~ASTNode() {
}

///////////////////////////////////////////////////////////////////////////
// SourcePos

SourcePos::SourcePos(const char *n, int l, int c) {
    name = n ? n : m->module->getModuleIdentifier().c_str();
    first_line = last_line = l;
    first_column = last_column = c;
}

llvm::DIFile SourcePos::GetDIFile() const {
#ifdef LLVM_2_8
    return llvm::DIFile();
#else
    std::string directory, filename;
    GetDirectoryAndFileName(g->currentDirectory, name, &directory, &filename);
    return m->diBuilder->createFile(filename, directory);
#endif // LLVM_2_8
}


void
SourcePos::Print() const { 
    printf(" @ [%s:%d.%d - %d.%d] ", name, first_line, first_column,
           last_line, last_column); 
}


bool
SourcePos::operator==(const SourcePos &p2) const {
    return (!strcmp(name, p2.name) && 
            first_line == p2.first_line &&
            first_column == p2.first_column &&
            last_line == p2.last_line &&
            last_column == p2.last_column);
}

