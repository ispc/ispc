/*
  Copyright (c) 2011-2022, Intel Corporation
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

/** @file func.h
    @brief Representation of a function in a source file.
*/

#pragma once

#include "ast.h"
#include "ispc.h"

#include <vector>

namespace ispc {

class Function {
  public:
    Function(Symbol *sym, Stmt *code);

    const Type *GetReturnType() const;
    const FunctionType *GetType() const;

    /** Generate LLVM IR for the function into the current module. */
    void GenerateIR();

    void Print() const;
    void Print(Indent &indent) const;
    bool IsStdlibSymbol() const;

  private:
    enum class DebugPrintPoint { Initial, AfterTypeChecking, AfterOptimization };
    void debugPrintHelper(DebugPrintPoint dumpPoint);

    void emitCode(FunctionEmitContext *ctx, llvm::Function *function, SourcePos firstStmtPos);

    Symbol *sym;
    std::vector<Symbol *> args;
    Stmt *code;
    Symbol *maskSymbol;
    Symbol *threadIndexSym, *threadCountSym;
    Symbol *taskIndexSym, *taskCountSym;
    Symbol *taskIndexSym0, *taskCountSym0;
    Symbol *taskIndexSym1, *taskCountSym1;
    Symbol *taskIndexSym2, *taskCountSym2;
};
} // namespace ispc
