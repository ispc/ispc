/*
  Copyright (c) 2011-2016, Intel Corporation
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

#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2 // 3.2
#ifdef ISPC_NVPTX_ENABLED
#include <llvm/Metadata.h>
#endif /* ISPC_NVPTX_ENABLED */
#include <llvm/DerivedTypes.h>
#include <llvm/Instructions.h>
#include <llvm/Intrinsics.h>
#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/Type.h>
#else
#ifdef ISPC_NVPTX_ENABLED
#include <llvm/IR/Metadata.h>
#endif /* ISPC_NVPTX_ENABLED */
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#endif
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_6
#include "llvm/PassManager.h"
#else // LLVM 3.7+
#include "llvm/IR/LegacyPassManager.h"
#endif
#include <llvm/PassRegistry.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO.h>
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_5 // LLVM 3.5+
#include <llvm/IR/CFG.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/Verifier.h>
#else
#include <llvm/Analysis/Verifier.h>
#include <llvm/Assembly/PrintModulePass.h>
#include <llvm/Support/CFG.h>
#endif
#include <llvm/Support/ToolOutputFile.h>

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
        code->Print(0);
        printf("\n\n\n");
    }

    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != NULL);

    for (int i = 0; i < type->GetNumParameters(); ++i) {
        const char *paramName = type->GetParameterName(i).c_str();
        Symbol *sym = m->symbolTable->LookupVariable(paramName);
        if (sym == NULL)
            Assert(strncmp(paramName, "__anon_parameter_", 17) == 0);
        args.push_back(sym);

        const Type *t = type->GetParameterType(i);
        if (sym != NULL && CastType<ReferenceType>(t) == NULL)
            sym->parentFunction = this;
    }

    if (type->isTask
#ifdef ISPC_NVPTX_ENABLED
        && (g->target->getISA() != Target::NVPTX)
#endif
    ) {
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
    Assert(llvm::isa<llvm::StructType>(pt->getElementType()));
    const llvm::StructType *argStructType = llvm::dyn_cast<const llvm::StructType>(pt->getElementType());

    // Get the type of the argument we're copying in and its Symbol pointer
    llvm::Type *argType = argStructType->getElementType(i);
    Symbol *sym = args[i];

    if (sym == NULL)
        // anonymous parameter, so don't worry about it
        return;

    // allocate space to copy the parameter in to
    sym->storagePtr = ctx->AllocaInst(argType, sym->name.c_str());

    // get a pointer to the value in the struct
    llvm::Value *ptr = ctx->AddElementOffset(structArgPtr, i, NULL, sym->name.c_str());

    // and copy the value from the struct and into the local alloca'ed
    // memory
    llvm::Value *ptrval = ctx->LoadInst(ptr, sym->name.c_str());
    ctx->StoreInst(ptrval, sym->storagePtr);
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

#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_7 // LLVM 3.7+
    if (g->NoOmitFramePointer)
        function->addFnAttr("no-frame-pointer-elim", "true");
#endif

#if 0
    llvm::BasicBlock *entryBBlock = ctx->GetCurrentBasicBlock();
#endif
    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != NULL);
    if (type->isTask == true
#ifdef ISPC_NVPTX_ENABLED
        && (g->target->getISA() != Target::NVPTX)
#endif
    ) {
        // For tasks, there should always be three parameters: the
        // pointer to the structure that holds all of the arguments, the
        // thread index, and the thread count variables.
        llvm::Function::arg_iterator argIter = function->arg_begin();
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
        llvm::Value *structParamPtr = argIter++;
        llvm::Value *threadIndex = argIter++;
        llvm::Value *threadCount = argIter++;
        llvm::Value *taskIndex = argIter++;
        llvm::Value *taskCount = argIter++;
        llvm::Value *taskIndex0 = argIter++;
        llvm::Value *taskIndex1 = argIter++;
        llvm::Value *taskIndex2 = argIter++;
        llvm::Value *taskCount0 = argIter++;
        llvm::Value *taskCount1 = argIter++;
        llvm::Value *taskCount2 = argIter++;
#else /* LLVM 3.8+ */
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
#endif
        // Copy the function parameter values from the structure into local
        // storage
        for (unsigned int i = 0; i < args.size(); ++i)
            lCopyInTaskParameter(i, structParamPtr, args, ctx);

        if (type->isUnmasked == false) {
            // Copy in the mask as well.
            int nArgs = (int)args.size();
            // The mask is the last parameter in the argument structure
            llvm::Value *ptr = ctx->AddElementOffset(structParamPtr, nArgs, NULL, "task_struct_mask");
            llvm::Value *ptrval = ctx->LoadInst(ptr, "mask");
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
        // Regular, non-task function
        llvm::Function::arg_iterator argIter = function->arg_begin();
        for (unsigned int i = 0; i < args.size(); ++i, ++argIter) {
            Symbol *sym = args[i];
            if (sym == NULL)
                // anonymous function parameter
                continue;

            argIter->setName(sym->name.c_str());

            // Allocate stack storage for the parameter and emit code
            // to store the its value there.
            sym->storagePtr = ctx->AllocaInst(argIter->getType(), sym->name.c_str());
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
            ctx->StoreInst(argIter, sym->storagePtr);
#else /* LLVM 3.8+ */
            ctx->StoreInst(&*argIter, sym->storagePtr);
#endif
            ctx->EmitFunctionParameterDebugInfo(sym, i);
        }

        // If the number of actual function arguments is equal to the
        // number of declared arguments in decl->functionParams, then we
        // don't have a mask parameter, so set it to be all on.  This
        // happens for exmaple with 'export'ed functions that the app
        // calls.
        if (argIter == function->arg_end()) {
            Assert(type->isUnmasked || type->isExported);
            ctx->SetFunctionMask(LLVMMaskAllOn);
        } else {
            Assert(type->isUnmasked == false);

            // Otherwise use the mask to set the entry mask value
            argIter->setName("__mask");
            Assert(argIter->getType() == LLVMTypes::MaskType);
#if ISPC_LLVM_VERSION <= ISPC_LLVM_3_7 /* 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 */
            ctx->SetFunctionMask(argIter);
#else /* LLVM 3.8+ */
            ctx->SetFunctionMask(&*argIter);
#endif
            Assert(++argIter == function->arg_end());
        }
#ifdef ISPC_NVPTX_ENABLED
        if (type->isTask == true && g->target->getISA() == Target::NVPTX) {
            llvm::NamedMDNode *annotations = m->module->getOrInsertNamedMetadata("nvvm.annotations");
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_6 // LLVM 3.6+
            llvm::SmallVector<llvm::Metadata *, 3> av;
            av.push_back(llvm::ValueAsMetadata::get(function));
            av.push_back(llvm::MDString::get(*g->ctx, "kernel"));
            av.push_back(llvm::ConstantAsMetadata::get(LLVMInt32(1)));
            annotations->addOperand(llvm::MDNode::get(*g->ctx, llvm::ArrayRef<llvm::Metadata *>(av)));
#else
            llvm::SmallVector<llvm::Value *, 3> av;
            av.push_back(function);
            av.push_back(llvm::MDString::get(*g->ctx, "kernel"));
            av.push_back(LLVMInt32(1));
            annotations->addOperand(llvm::MDNode::get(*g->ctx, av));
#endif
        }
#endif /* ISPC_NVPTX_ENABLED */
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
        bool checkMask = (type->isTask == true) ||
                         (
#if ISPC_LLVM_VERSION == ISPC_LLVM_3_2 // 3.2
                             (function->getFnAttributes().hasAttribute(llvm::Attributes::AlwaysInline) == false)
#elif ISPC_LLVM_VERSION <= ISPC_LLVM_4_0
                             (function->getAttributes().getFnAttributes().hasAttribute(
                                  llvm::AttributeSet::FunctionIndex, llvm::Attribute::AlwaysInline) == false)
#else // LLVM 5.0+
                             (function->getAttributes().getFnAttributes().hasAttribute(llvm::Attribute::AlwaysInline) ==
                              false)
#endif
                             && costEstimate > CHECK_MASK_AT_FUNCTION_START_COST);
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
    {
        FunctionEmitContext ec(this, sym, function, firstStmtPos);
        emitCode(&ec, function, firstStmtPos);
    }

    if (m->errorCount == 0) {
        // If the function is 'export'-qualified, emit a second version of
        // it without a mask parameter and without name mangling so that
        // the application can call it
        const FunctionType *type = CastType<FunctionType>(sym->type);
        Assert(type != NULL);
        if (type->isExported) {
            if (!type->isTask) {
                llvm::FunctionType *ftype = type->LLVMFunctionType(g->ctx, true);
                llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::ExternalLinkage;
                std::string functionName = sym->name;
                if (g->mangleFunctionsWithTarget) {
                    // If we treat generic as smth, we should have appropriate mangling
                    if (g->target->getISA() == Target::GENERIC && !g->target->getTreatGenericAsSmth().empty())
                        functionName += std::string("_") + g->target->getTreatGenericAsSmth();
                    else
                        functionName += std::string("_") + g->target->GetISAString();
                }
#ifdef ISPC_NVPTX_ENABLED
                if (g->target->getISA() == Target::NVPTX) {
                    functionName +=
                        std::string("___export"); /* add ___export to the end, for ptxcc to recognize it is exported */
#if 0
                  llvm::NamedMDNode* annotations =
                    m->module->getOrInsertNamedMetadata("nvvm.annotations");
                  llvm::SmallVector<llvm::Value*, 3> av;
                  av.push_back(function);
                  av.push_back(llvm::MDString::get(*g->ctx, "kernel"));
                  av.push_back(llvm::ConstantInt::get(llvm::IntegerType::get(*g->ctx,32), 1));
                  annotations->addOperand(llvm::MDNode::get(*g->ctx, av));
#endif
                }
#endif /* ISPC_NVPTX_ENABLED */
                llvm::Function *appFunction = llvm::Function::Create(ftype, linkage, functionName.c_str(), m->module);
                appFunction->setDoesNotThrow();

#if ISPC_LLVM_VERSION < ISPC_LLVM_5_0
                // We should iterate from 1 because zero parameter is return.
                // We should iterate till getNumParams instead of getNumParams+1 because new
                // function is export function and doesn't contain the last parameter "mask".
                for (int i = 1; i < function->getFunctionType()->getNumParams(); i++) {
                    if (function->doesNotAlias(i)) {
                        appFunction->setDoesNotAlias(i);
                    }
                }
#else // LLVM 5.0+
                for (int i = 0; i < function->getFunctionType()->getNumParams() - 1; i++) {
                    if (function->hasParamAttribute(i, llvm::Attribute::NoAlias)) {
                        appFunction->addParamAttr(i, llvm::Attribute::NoAlias);
                    }
                }
#endif
                g->target->markFuncWithTargetAttr(appFunction);

                if (appFunction->getName() != functionName) {
                    // this was a redefinition for which we already emitted an
                    // error, so don't worry about this one...
                    appFunction->eraseFromParent();
                } else {
                    // And emit the code again
                    FunctionEmitContext ec(this, sym, appFunction, firstStmtPos);
                    emitCode(&ec, appFunction, firstStmtPos);
                    if (m->errorCount == 0) {
                        sym->exportedFunction = appFunction;
                    }
#ifdef ISPC_NVPTX_ENABLED
                    if (g->target->getISA() == Target::NVPTX) {
                        llvm::NamedMDNode *annotations = m->module->getOrInsertNamedMetadata("nvvm.annotations");
#if ISPC_LLVM_VERSION >= ISPC_LLVM_3_6 // LLVM 3.6+

                        llvm::SmallVector<llvm::Metadata *, 3> av;
                        av.push_back(llvm::ValueAsMetadata::get(appFunction));
                        av.push_back(llvm::MDString::get(*g->ctx, "kernel"));
                        av.push_back(llvm::ConstantAsMetadata::get(
                            llvm::ConstantInt::get(llvm::IntegerType::get(*g->ctx, 32), 1)));
                        annotations->addOperand(llvm::MDNode::get(*g->ctx, llvm::ArrayRef<llvm::Metadata *>(av)));
#else
                        llvm::SmallVector<llvm::Value *, 3> av;
                        av.push_back(appFunction);
                        av.push_back(llvm::MDString::get(*g->ctx, "kernel"));
                        av.push_back(llvm::ConstantInt::get(llvm::IntegerType::get(*g->ctx, 32), 1));
                        annotations->addOperand(llvm::MDNode::get(*g->ctx, av));
#endif
                    }
#endif /* ISPC_NVPTX_ENABLED */
                }
            }
        }
    }
}
