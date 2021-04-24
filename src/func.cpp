/*
  Copyright (c) 2011-2021, Intel Corporation
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

/** @file func.cpp
    @brief
*/

#include "func.h"
#include "ctx.h"
#include "expr.h"
#include "llvmutil.h"
#include "module.h"
#include "stmt.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <stdio.h>

#include <llvm/IR/CFG.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO.h>

#ifdef ISPC_GENX_ENABLED
#include <llvm/GenXIntrinsics/GenXMetadata.h>
#endif

using namespace ispc;

Function::Function(Symbol *s, Stmt *c) {
    sym = s;
    code = c;

    maskSymbol = m->symbolTable->LookupVariable("__mask");
    Assert(maskSymbol != NULL);

    if (code != NULL) {
        code = TypeCheck(code);

        if (code != NULL && g->debugPrint) {
            printf("After typechecking function \"%s\":\n", sym->name.c_str());
            code->Print(0);
            printf("---------------------\n");
        }

        if (code != NULL) {
            code = Optimize(code);
            if (g->debugPrint) {
                printf("After optimizing function \"%s\":\n", sym->name.c_str());
                code->Print(0);
                printf("---------------------\n");
            }
        }
    }

    if (g->debugPrint) {
        printf("Add Function %s\n", sym->name.c_str());
        if (code != NULL) {
            code->Print(0);
        }
        printf("\n\n\n");
    }

    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != NULL);

    for (int i = 0; i < type->GetNumParameters(); ++i) {
        const char *paramName = type->GetParameterName(i).c_str();
        Symbol *paramSym = m->symbolTable->LookupVariable(paramName);
        if (paramSym == NULL)
            Assert(strncmp(paramName, "__anon_parameter_", 17) == 0);
        args.push_back(paramSym);

        const Type *t = type->GetParameterType(i);
        if (paramSym != NULL && CastType<ReferenceType>(t) == NULL)
            paramSym->parentFunction = this;
    }

    if (type->isTask && (!g->target->isGenXTarget())) {
        threadIndexSym = m->symbolTable->LookupVariable("threadIndex");
        Assert(threadIndexSym);
        threadCountSym = m->symbolTable->LookupVariable("threadCount");
        Assert(threadCountSym);
        taskIndexSym = m->symbolTable->LookupVariable("taskIndex");
        Assert(taskIndexSym);
        taskCountSym = m->symbolTable->LookupVariable("taskCount");
        Assert(taskCountSym);

        taskIndexSym0 = m->symbolTable->LookupVariable("taskIndex0");
        Assert(taskIndexSym0);
        taskIndexSym1 = m->symbolTable->LookupVariable("taskIndex1");
        Assert(taskIndexSym1);
        taskIndexSym2 = m->symbolTable->LookupVariable("taskIndex2");
        Assert(taskIndexSym2);

        taskCountSym0 = m->symbolTable->LookupVariable("taskCount0");
        Assert(taskCountSym0);
        taskCountSym1 = m->symbolTable->LookupVariable("taskCount1");
        Assert(taskCountSym1);
        taskCountSym2 = m->symbolTable->LookupVariable("taskCount2");
        Assert(taskCountSym2);
    } else {
        threadIndexSym = threadCountSym = taskIndexSym = taskCountSym = NULL;
        taskIndexSym0 = taskIndexSym1 = taskIndexSym2 = NULL;
        taskCountSym0 = taskCountSym1 = taskCountSym2 = NULL;
    }
}

const Type *Function::GetReturnType() const {
    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != NULL);
    return type->GetReturnType();
}

const FunctionType *Function::GetType() const {
    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != NULL);
    return type;
}

/** Parameters for tasks are stored in a big structure; this utility
    function emits code to copy those values out of the task structure into
    local stack-allocated variables.  (Which we expect that LLVM's
    'mem2reg' pass will in turn promote to SSA registers..
 */
static void lCopyInTaskParameter(int i, llvm::Value *structArgPtr, const std::vector<Symbol *> &args,
                                 FunctionEmitContext *ctx) {
    // We expect the argument structure to come in as a poitner to a
    // structure.  Confirm and figure out its type here.
    const llvm::Type *structArgType = structArgPtr->getType();
    Assert(llvm::isa<llvm::PointerType>(structArgType));
    const llvm::PointerType *pt = llvm::dyn_cast<const llvm::PointerType>(structArgType);
    Assert(pt);
    Assert(llvm::isa<llvm::StructType>(pt->getElementType()));

    // Get the type of the argument we're copying in and its Symbol pointer
    Symbol *sym = args[i];

    if (sym == NULL)
        // anonymous parameter, so don't worry about it
        return;

    // allocate space to copy the parameter in to
    sym->storagePtr = ctx->AllocaInst(sym->type, sym->name.c_str());

    // get a pointer to the value in the struct
    llvm::Value *ptr = ctx->AddElementOffset(structArgPtr, i, NULL, sym->name.c_str());

    // and copy the value from the struct and into the local alloca'ed
    // memory
    llvm::Value *ptrval = ctx->LoadInst(ptr, sym->type, sym->name.c_str());
    ctx->StoreInst(ptrval, sym->storagePtr, sym->type, sym->type->IsUniformType());
    ctx->EmitFunctionParameterDebugInfo(sym, i);
}

/** Given the statements implementing a function, emit the code that
    implements the function.  Most of the work do be done here just
    involves wiring up the function parameter values to be available in the
    function body code.
 */
void Function::emitCode(FunctionEmitContext *ctx, llvm::Function *function, SourcePos firstStmtPos) {
    // Connect the __mask builtin to the location in memory that stores its
    // value
    maskSymbol->storagePtr = ctx->GetFullMaskPointer();

    // add debugging info for __mask
    maskSymbol->pos = firstStmtPos;
    ctx->EmitVariableDebugInfo(maskSymbol);

    if (g->NoOmitFramePointer)
        function->addFnAttr("no-frame-pointer-elim", "true");
    if (g->target->getArch() == Arch::wasm32)
        function->addFnAttr("target-features", "+simd128");

    g->target->markFuncWithTargetAttr(function);
#if 0
    llvm::BasicBlock *entryBBlock = ctx->GetCurrentBasicBlock();
#endif
    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != NULL);
    if (type->isTask == true && (!g->target->isGenXTarget())) {
        // For tasks, there should always be three parameters: the
        // pointer to the structure that holds all of the arguments, the
        // thread index, and the thread count variables.
        llvm::Function::arg_iterator argIter = function->arg_begin();

        llvm::Value *structParamPtr = &*(argIter++);
        llvm::Value *threadIndex = &*(argIter++);
        llvm::Value *threadCount = &*(argIter++);
        llvm::Value *taskIndex = &*(argIter++);
        llvm::Value *taskCount = &*(argIter++);
        llvm::Value *taskIndex0 = &*(argIter++);
        llvm::Value *taskIndex1 = &*(argIter++);
        llvm::Value *taskIndex2 = &*(argIter++);
        llvm::Value *taskCount0 = &*(argIter++);
        llvm::Value *taskCount1 = &*(argIter++);
        llvm::Value *taskCount2 = &*(argIter++);
        // Copy the function parameter values from the structure into local
        // storage
        for (unsigned int i = 0; i < args.size(); ++i)
            lCopyInTaskParameter(i, structParamPtr, args, ctx);

        if (type->isUnmasked == false) {
            // Copy in the mask as well.
            int nArgs = (int)args.size();
            // The mask is the last parameter in the argument structure
            llvm::Value *ptr = ctx->AddElementOffset(structParamPtr, nArgs, NULL, "task_struct_mask");
            llvm::Value *ptrval = ctx->LoadInst(ptr, NULL, "mask");
            ctx->SetFunctionMask(ptrval);
        }

        // Copy threadIndex and threadCount into stack-allocated storage so
        // that their symbols point to something reasonable.
        threadIndexSym->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "threadIndex");
        ctx->StoreInst(threadIndex, threadIndexSym->storagePtr);

        threadCountSym->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "threadCount");
        ctx->StoreInst(threadCount, threadCountSym->storagePtr);

        // Copy taskIndex and taskCount into stack-allocated storage so
        // that their symbols point to something reasonable.
        taskIndexSym->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex");
        ctx->StoreInst(taskIndex, taskIndexSym->storagePtr);

        taskCountSym->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount");
        ctx->StoreInst(taskCount, taskCountSym->storagePtr);

        taskIndexSym0->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex0");
        ctx->StoreInst(taskIndex0, taskIndexSym0->storagePtr);
        taskIndexSym1->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex1");
        ctx->StoreInst(taskIndex1, taskIndexSym1->storagePtr);
        taskIndexSym2->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex2");
        ctx->StoreInst(taskIndex2, taskIndexSym2->storagePtr);

        taskCountSym0->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount0");
        ctx->StoreInst(taskCount0, taskCountSym0->storagePtr);
        taskCountSym1->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount1");
        ctx->StoreInst(taskCount1, taskCountSym1->storagePtr);
        taskCountSym2->storagePtr = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount2");
        ctx->StoreInst(taskCount2, taskCountSym2->storagePtr);
    } else {
        // Regular, non-task function or GPU task
        llvm::Function::arg_iterator argIter = function->arg_begin();
        for (unsigned int i = 0; i < args.size(); ++i, ++argIter) {
            Symbol *argSym = args[i];
            if (argSym == NULL)
                // anonymous function parameter
                continue;

            argIter->setName(argSym->name.c_str());

            // Allocate stack storage for the parameter and emit code
            // to store the its value there.
            argSym->storagePtr = ctx->AllocaInst(argSym->type, argSym->name.c_str());

            ctx->StoreInst(&*argIter, argSym->storagePtr, argSym->type);
            ctx->EmitFunctionParameterDebugInfo(argSym, i);
        }

        // If the number of actual function arguments is equal to the
        // number of declared arguments in decl->functionParams, then we
        // don't have a mask parameter, so set it to be all on.  This
        // happens for exmaple with 'export'ed functions that the app
        // calls.
        if (argIter == function->arg_end()) {
            Assert(type->isUnmasked || type->isExported || (g->target->isGenXTarget() && type->isTask));
            ctx->SetFunctionMask(LLVMMaskAllOn);
        } else {
            Assert(type->isUnmasked == false);

            // Otherwise use the mask to set the entry mask value
            argIter->setName("__mask");
            Assert(argIter->getType() == LLVMTypes::MaskType);

            if (ctx->emitGenXHardwareMask()) {
                // We should not create explicit predication
                // to avoid EM usage duplication. All stuff
                // will be done by SIMD CF Lowering
                // TODO: temporary workaround that will be changed
                // as part of SPIR-V emitting solution
                ctx->SetFunctionMask(LLVMMaskAllOn);
            } else {
                ctx->SetFunctionMask(&*argIter);
            }

            Assert(++argIter == function->arg_end());
        }
    }

    // Finally, we can generate code for the function
    if (code != NULL) {
        ctx->SetDebugPos(code->pos);
        ctx->AddInstrumentationPoint("function entry");

        int costEstimate = EstimateCost(code);
        Debug(code->pos, "Estimated cost for function \"%s\" = %d\n", sym->name.c_str(), costEstimate);

        // If the body of the function is non-trivial, then we wrap the
        // entire thing inside code that tests to see if the mask is all
        // on, all off, or mixed.  If this is a simple function, then this
        // isn't worth the code bloat / overhead.
        bool checkMask =
            (!g->target->isGenXTarget() && type->isTask == true) ||
            ((function->getAttributes().getFnAttributes().hasAttribute(llvm::Attribute::AlwaysInline) == false) &&
             costEstimate > CHECK_MASK_AT_FUNCTION_START_COST);
        checkMask &= (type->isUnmasked == false);
        checkMask &= (g->target->getMaskingIsFree() == false);
        checkMask &= (g->opt.disableCoherentControlFlow == false);

        if (checkMask) {
            llvm::Value *mask = ctx->GetFunctionMask();
            llvm::Value *allOn = ctx->All(mask);
            llvm::BasicBlock *bbAllOn = ctx->CreateBasicBlock("all_on");
            llvm::BasicBlock *bbSomeOn = ctx->CreateBasicBlock("some_on");

            // Set up basic blocks for goto targets
            ctx->InitializeLabelMap(code);

            ctx->BranchInst(bbAllOn, bbSomeOn, allOn);
            // all on: we've determined dynamically that the mask is all
            // on.  Set the current mask to "all on" explicitly so that
            // codegen for this path can be improved with this knowledge in
            // hand...
            ctx->SetCurrentBasicBlock(bbAllOn);
            if (!g->opt.disableMaskAllOnOptimizations)
                ctx->SetFunctionMask(LLVMMaskAllOn);
            code->EmitCode(ctx);
            if (ctx->GetCurrentBasicBlock())
                ctx->ReturnInst();

            // not all on: however, at least one lane must be running,
            // since we should never run with all off...  some on: reset
            // the mask to the value it had at function entry and emit the
            // code.  Resetting the mask here is important, due to the "all
            // on" setting of it for the path above.
            ctx->SetCurrentBasicBlock(bbSomeOn);
            ctx->SetFunctionMask(mask);

            // Set up basic blocks for goto targets again; we want to have
            // one set of them for gotos in the 'all on' case, and a
            // distinct set for the 'mixed mask' case.
            ctx->InitializeLabelMap(code);

            code->EmitCode(ctx);
            if (ctx->GetCurrentBasicBlock())
                ctx->ReturnInst();
        } else {
            // Set up basic blocks for goto targets
            ctx->InitializeLabelMap(code);
            // No check, just emit the code
            code->EmitCode(ctx);
        }
    }

    if (ctx->GetCurrentBasicBlock()) {
        // FIXME: We'd like to issue a warning if we've reached the end of
        // the function without a return statement (for non-void
        // functions).  But the test below isn't right, since we can have
        // (with 'x' a varying test) "if (x) return a; else return b;", in
        // which case we have a valid basic block but its unreachable so ok
        // to not have return statement.
#if 0
        // If the bblock has no predecessors, then it doesn't matter if it
        // doesn't have a return; it'll never be reached.  If it does,
        // issue a warning.  Also need to warn if it's the entry block for
        // the function (in which case it will not have predeccesors but is
        // still reachable.)
        if (type->GetReturnType()->IsVoidType() == false &&
            (pred_begin(ec.bblock) != pred_end(ec.bblock) || (ec.bblock == entryBBlock)))
            Warning(sym->pos, "Missing return statement in function returning \"%s\".",
                    type->rType->GetString().c_str());
#endif

        // FIXME: would like to set the context's current position to
        // e.g. the end of the function code

        // if bblock is non-NULL, it hasn't been terminated by e.g. a
        // return instruction.  Need to add a return instruction.
        ctx->ReturnInst();
    }
#ifdef ISPC_GENX_ENABLED
    if (g->target->isGenXTarget()) {
        // Emit metadata for GENX kernel
        if (type->isExported || type->isTask) {
            llvm::LLVMContext &fContext = function->getContext();
            llvm::NamedMDNode *mdKernels = m->module->getOrInsertNamedMetadata("genx.kernels");

            std::string AsmName = (m->module->getName() + llvm::Twine('_') + llvm::Twine(mdKernels->getNumOperands()) +
                                   llvm::Twine(".asm"))
                                      .str();

            // Kernel arg kinds
            llvm::Type *i32Type = llvm::Type::getInt32Ty(fContext);
            llvm::SmallVector<llvm::Metadata *, 8> argKinds;
            llvm::SmallVector<llvm::Metadata *, 8> argInOutKinds;
            llvm::SmallVector<llvm::Metadata *, 8> argOffsets;
            llvm::SmallVector<llvm::Metadata *, 8> argTypeDescs;

            // In ISPC we need only AK_NORMAL and IK_NORMAL now, in future it can change.
            enum { AK_NORMAL, AK_SAMPLER, AK_SURFACE, AK_VME };
            enum { IK_NORMAL, IK_INPUT, IK_OUTPUT, IK_INPUT_OUTPUT };
            unsigned int offset = 32;
            unsigned int grf_size = g->target->getGenxGrfSize();
            for (int i = 0; i < args.size(); i++) {
                const Type *T = args[i]->type;
                argKinds.push_back(llvm::ValueAsMetadata::get(llvm::ConstantInt::get(i32Type, AK_NORMAL)));
                argInOutKinds.push_back(llvm::ValueAsMetadata::get(llvm::ConstantInt::get(i32Type, IK_NORMAL)));
                llvm::Type *argType = function->getArg(i)->getType();
                if (argType->isPtrOrPtrVectorTy() || argType->isArrayTy()) {
                    argTypeDescs.push_back(llvm::MDString::get(fContext, llvm::StringRef("svmptr_t read_write")));
                } else {
                    argTypeDescs.push_back(llvm::MDString::get(fContext, llvm::StringRef("")));
                }

                llvm::Type *type = T->LLVMType(&fContext);
                unsigned bytes = type->getScalarSizeInBits() / 8;
                if (bytes != 0) {
                    offset = llvm::alignTo(offset, bytes);
                }

                if (llvm::isa<llvm::VectorType>(type)) {
                    bytes = type->getPrimitiveSizeInBits() / 8;

                    if ((offset & (grf_size - 1)) + bytes > grf_size)
                        // GRF align if arg would cross GRF boundary
                        offset = llvm::alignTo(offset, grf_size);
                }

                argOffsets.push_back(llvm::ValueAsMetadata::get(llvm::ConstantInt::get(i32Type, offset)));

                offset += bytes;
            }

            // TODO: Number of fields is 9 now, and it is a magic number that seems
            // to be not defined anywhere. Consider changing it when possible.
            llvm::SmallVector<llvm::Metadata *, 9> mdArgs(9, nullptr);
            mdArgs[llvm::genx::KernelMDOp::FunctionRef] = llvm::ValueAsMetadata::get(function);
            mdArgs[llvm::genx::KernelMDOp::Name] = llvm::MDString::get(fContext, sym->name);
            mdArgs[llvm::genx::KernelMDOp::ArgKinds] = llvm::MDNode::get(fContext, argKinds);
            mdArgs[llvm::genx::KernelMDOp::SLMSize] =
                llvm::ValueAsMetadata::get(llvm::ConstantInt::getNullValue(i32Type));
            mdArgs[llvm::genx::KernelMDOp::ArgOffsets] =
                llvm::ValueAsMetadata::get(llvm::ConstantInt::getNullValue(i32Type));
            mdArgs[llvm::genx::KernelMDOp::ArgIOKinds] = llvm::MDNode::get(fContext, argInOutKinds);
            mdArgs[llvm::genx::KernelMDOp::ArgTypeDescs] = llvm::MDNode::get(fContext, argTypeDescs);
            mdArgs[llvm::genx::KernelMDOp::Reserved_0] =
                llvm::ValueAsMetadata::get(llvm::ConstantInt::getNullValue(i32Type));
            mdArgs[llvm::genx::KernelMDOp::BarrierCnt] =
                llvm::ValueAsMetadata::get(llvm::ConstantInt::getNullValue(i32Type));

            mdKernels->addOperand(llvm::MDNode::get(fContext, mdArgs));
            // This is needed to run in L0 runtime.
            function->addFnAttr("oclrt", "1");
        }
    }
#endif
}

void Function::GenerateIR() {
    if (sym == NULL)
        // May be NULL due to error earlier in compilation
        return;

    llvm::Function *function = sym->function;
    Assert(function != NULL);

    // But if that function has a definition, we don't want to redefine it.
    if (function->empty() == false) {
        Error(sym->pos, "Ignoring redefinition of function \"%s\".", sym->name.c_str());
        return;
    }

    // If function is an 'extern C', it cannot be defined in ISPC.
    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != NULL);
    if (type->isExternC) {
        Error(sym->pos, "\n\'extern \"C\"\' function \"%s\" cannot be defined in ISPC.", sym->name.c_str());
        return;
    }

    // Figure out a reasonable source file position for the start of the
    // function body.  If possible, get the position of the first actual
    // non-StmtList statment...
    SourcePos firstStmtPos = sym->pos;
    if (code) {
        StmtList *sl = llvm::dyn_cast<StmtList>(code);
        if (sl && sl->stmts.size() > 0 && sl->stmts[0] != NULL)
            firstStmtPos = sl->stmts[0]->pos;
        else
            firstStmtPos = code->pos;
    }
    // And we can now go ahead and emit the code
    if (g->target->isGenXTarget()) {
        // For GEN target we emit code only for unmasked version of a kernel and subroutines.
        // TODO_GEN: revise this one more time after testing of subroutines calls.
        const FunctionType *type = CastType<FunctionType>(sym->type);
        if ((type->isExported && type->isUnmasked) || (!type->isExported && !type->isTask)) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_10_0
            llvm::TimeTraceScope TimeScope("emitCode", llvm::StringRef(sym->name));
#endif
            FunctionEmitContext ec(this, sym, function, firstStmtPos);
            emitCode(&ec, function, firstStmtPos);
        }
    } else {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_10_0
        llvm::TimeTraceScope TimeScope("emitCode", llvm::StringRef(sym->name));
#endif
        FunctionEmitContext ec(this, sym, function, firstStmtPos);
        emitCode(&ec, function, firstStmtPos);
    }

    if (m->errorCount == 0) {
        // If the function is 'export'-qualified, emit a second version of
        // it without a mask parameter and without name mangling so that
        // the application can call it
        // For gen we emit a version without mask parameter only for "export" -qualified functions and tasks.
        if (type->isExported || (g->target->isGenXTarget() && type->isTask)) {
            if ((!g->target->isGenXTarget() && !type->isTask) || g->target->isGenXTarget()) {
                llvm::FunctionType *ftype = type->LLVMFunctionType(g->ctx, true);
                llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::ExternalLinkage;
                std::string functionName = sym->name;
                if (g->mangleFunctionsWithTarget) {
                    functionName += std::string("_") + g->target->GetISAString();
                }

                llvm::Function *appFunction = llvm::Function::Create(ftype, linkage, functionName.c_str(), m->module);
                appFunction->setDoesNotThrow();
                g->target->markFuncWithCallingConv(appFunction);

                // GenX kernel should have "dllexport" and "CMGenxMain" attribute
                if (g->target->isGenXTarget()) {
                    appFunction->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
                    appFunction->addFnAttr("CMGenxMain");
                }

                for (int i = 0; i < function->getFunctionType()->getNumParams() - 1; i++) {
                    if (function->hasParamAttribute(i, llvm::Attribute::NoAlias)) {
                        appFunction->addParamAttr(i, llvm::Attribute::NoAlias);
                    }
                }
                g->target->markFuncWithTargetAttr(appFunction);

                if (appFunction->getName() != functionName) {
                    // this was a redefinition for which we already emitted an
                    // error, so don't worry about this one...
                    appFunction->eraseFromParent();
                } else {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_10_0
                    llvm::TimeTraceScope TimeScope("emitCode", llvm::StringRef(sym->name));
#endif
                    // And emit the code again
                    FunctionEmitContext ec(this, sym, appFunction, firstStmtPos);
                    emitCode(&ec, appFunction, firstStmtPos);
                    if (m->errorCount == 0) {
                        sym->exportedFunction = appFunction;
                    }
                }
            }
        } else {
            // In case if it is not the kernel, mark function as a stack call
            if (g->target->isGenXTarget()) {
                function->addFnAttr("CMStackCall");
            }
        }
    }
}
