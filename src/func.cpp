/*
  Copyright (c) 2011-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file func.cpp
    @brief
*/

#include "func.h"
#include "builtins-decl.h"
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

#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXMetadata.h>
#endif

using namespace ispc;

///////////////////////////////////////////////////////////////////////////
// Function

bool Function::IsStdlibSymbol() const {
    if (sym == nullptr) {
        return false;
    }

    if (sym->pos.name != nullptr && !strcmp(sym->pos.name, "stdlib.ispc")) {
        return true;
    }
    return false;
}

void Function::debugPrintHelper(DebugPrintPoint dumpPoint) {
    if (code == nullptr || sym == nullptr) {
        return;
    }

    if (!g->debugPrint) {
        return;
    }

    // With debug prints enabled we will dump AST on several stages, so need annotation.
    if (g->debugPrint) {
        switch (dumpPoint) {
        case DebugPrintPoint::Initial:
            printf("Initial AST\n");
            break;
        case DebugPrintPoint::AfterTypeChecking:
            printf("AST after after typechecking\n");
            break;
        case DebugPrintPoint::AfterOptimization:
            printf("AST after optimization\n");
            break;
        }
    }

    Print();
    printf("\n");
}

void Function::Print() const {
    Indent indent;
    indent.pushSingle();
    Print(indent);
    fflush(stdout);
}

void Function::Print(Indent &indent) const {
    indent.Print("Function");

    if (sym && sym->type) {
        sym->pos.Print();
        printf(" [%s] \"%s\"\n", sym->type->GetString().c_str(), sym->name.c_str());
    } else {
        printf("<NULL>\n");
    }

    indent.pushList(args.size() + 1);
    for (int i = 0; i < args.size(); i++) {
        static constexpr std::size_t BUFSIZE{15};
        char buffer[BUFSIZE];
        snprintf(buffer, BUFSIZE, "param %d", i);
        indent.setNextLabel(buffer);
        if (args[i]) {
            indent.Print();
            if (args[i]->type != nullptr) {
                printf("[%s] ", args[i]->type->GetString().c_str());
            }
            printf("%s\n", args[i]->name.c_str());
            indent.Done();
        } else {
            indent.Print("<NULL>\n");
            indent.Done();
        }
    }

    indent.setNextLabel("body");
    if (code != nullptr) {
        code->Print(indent);
    } else {
        printf("<CODE is missing>\n");
    }
    indent.Done();
}

// The Function is created when the body of the function is already parsed and AST is created for it,
// and we are about to close the symbol table scope for the function. So all symbols that require special
// handling during code generation must be saved. This includes symbols for arguments and special symbols
// like __mask and thread / task variables.
// Type checking and optimization is also done here.
Function::Function(Symbol *s, Stmt *c) : sym(s), code(c) {
    maskSymbol = m->symbolTable->LookupVariable("__mask");
    Assert(maskSymbol != nullptr);

    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != nullptr);

    for (int i = 0; i < type->GetNumParameters(); ++i) {
        const char *paramName = type->GetParameterName(i).c_str();
        Symbol *paramSym = m->symbolTable->LookupVariable(paramName);
        if (paramSym == nullptr)
            Assert(strncmp(paramName, "__anon_parameter_", 17) == 0);
        args.push_back(paramSym);

        const Type *t = type->GetParameterType(i);
        if (paramSym != nullptr && CastType<ReferenceType>(t) == nullptr)
            paramSym->parentFunction = this;
    }

    if (type->isTask) {
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
        threadIndexSym = threadCountSym = taskIndexSym = taskCountSym = nullptr;
        taskIndexSym0 = taskIndexSym1 = taskIndexSym2 = nullptr;
        taskCountSym0 = taskCountSym1 = taskCountSym2 = nullptr;
    }

    typeCheckAndOptimize();
}

// The version of constructor, which accepts symbols directly instead of doing lookup in the symbol table.
// This is necessary to instantiate template functions, as symbol lookup is not available during instantiation.
Function::Function(Symbol *s, Stmt *c, Symbol *ms, std::vector<Symbol *> &a)
    : sym(s), args(a), code(c), maskSymbol(ms), threadIndexSym(nullptr), threadCountSym(nullptr), taskIndexSym(nullptr),
      taskCountSym(nullptr), taskIndexSym0(nullptr), taskCountSym0(nullptr), taskIndexSym1(nullptr),
      taskCountSym1(nullptr), taskIndexSym2(nullptr), taskCountSym2(nullptr) {
    typeCheckAndOptimize();
}

void Function::typeCheckAndOptimize() {
    if (code != nullptr) {
        debugPrintHelper(DebugPrintPoint::Initial);

        code = TypeCheck(code);

        debugPrintHelper(DebugPrintPoint::AfterTypeChecking);

        if (code != nullptr) {
            code = Optimize(code);

            debugPrintHelper(DebugPrintPoint::AfterOptimization);
        }
    }
}

const Type *Function::GetReturnType() const {
    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != nullptr);
    return type->GetReturnType();
}

const FunctionType *Function::GetType() const {
    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != nullptr);
    return type;
}

/** Parameters for tasks are stored in a big structure; this utility
    function emits code to copy those values out of the task structure into
    local stack-allocated variables.  (Which we expect that LLVM's
    'mem2reg' pass will in turn promote to SSA registers..
 */
static void lCopyInTaskParameter(int i, AddressInfo *structArgPtrInfo, const std::vector<Symbol *> &args,
                                 FunctionEmitContext *ctx) {
    // We expect the argument structure to come in as a poitner to a
    // structure.  Confirm and figure out its type here.
    const llvm::Type *structArgType = structArgPtrInfo->getPointer()->getType();
    Assert(llvm::isa<llvm::PointerType>(structArgType));
    const llvm::PointerType *pt = llvm::dyn_cast<const llvm::PointerType>(structArgType);
    Assert(pt);
    Assert(llvm::isa<llvm::StructType>(structArgPtrInfo->getElementType()));

    // Get the type of the argument we're copying in and its Symbol pointer
    Symbol *sym = args[i];

    if (sym == nullptr)
        // anonymous parameter, so don't worry about it
        return;

    // allocate space to copy the parameter in to
    sym->storageInfo = ctx->AllocaInst(sym->type, sym->name.c_str());
    Assert(sym->storageInfo);

    // get a pointer to the value in the struct
    llvm::Value *ptr = ctx->AddElementOffset(structArgPtrInfo, i, sym->name.c_str());

    // and copy the value from the struct and into the local alloca'ed
    // memory
    llvm::Value *ptrval =
        ctx->LoadInst(new AddressInfo(ptr, sym->storageInfo->getElementType()), sym->type, sym->name.c_str());
    ctx->StoreInst(ptrval, sym->storageInfo, sym->type);
    ctx->EmitFunctionParameterDebugInfo(sym, i);
}

static llvm::Value *lXeGetTaskVariableValue(FunctionEmitContext *ctx, std::string taskFunc) {
    std::vector<llvm::Value *> args;
    llvm::Function *task_func = m->module->getFunction(taskFunc);
    Assert(task_func != nullptr);
    return ctx->CallInst(task_func, nullptr, args, taskFunc + "_call");
}

/** Given the statements implementing a function, emit the code that
    implements the function.  Most of the work do be done here just
    involves wiring up the function parameter values to be available in the
    function body code.
 */
void Function::emitCode(FunctionEmitContext *ctx, llvm::Function *function, SourcePos firstStmtPos) {
    // Connect the __mask builtin to the location in memory that stores its
    // value
    maskSymbol->storageInfo = ctx->GetFullMaskAddressInfo();

    // add debugging info for __mask
    maskSymbol->pos = firstStmtPos;
    ctx->EmitVariableDebugInfo(maskSymbol);

    if (g->NoOmitFramePointer)
        function->addFnAttr("frame-pointer", "all");
    if (g->target->getArch() == Arch::wasm32 || g->target->getArch() == Arch::wasm64)
        function->addFnAttr("target-features", "+simd128");

    g->target->markFuncWithTargetAttr(function);
    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != nullptr);

    // CPU tasks
    if (type->isTask == true && !g->target->isXeTarget()) {
        Assert(type->IsISPCExternal() == false);
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

        std::vector<llvm::Type *> llvmArgTypes = type->LLVMFunctionArgTypes(g->ctx);
        llvm::Type *st = llvm::StructType::get(*g->ctx, llvmArgTypes);
        AddressInfo *stInfo = new AddressInfo(structParamPtr, st);
        // Copy the function parameter values from the structure into local
        // storage
        for (unsigned int i = 0; i < args.size(); ++i)
            lCopyInTaskParameter(i, stInfo, args, ctx);

        if (type->isUnmasked == false) {
            // Copy in the mask as well.
            int nArgs = (int)args.size();
            // The mask is the last parameter in the argument structure
            llvm::Value *ptr = ctx->AddElementOffset(stInfo, nArgs, "task_struct_mask");
            llvm::Value *ptrval = ctx->LoadInst(new AddressInfo(ptr, LLVMTypes::MaskType), nullptr, "mask");
            ctx->SetFunctionMask(ptrval);
        }

        // Copy threadIndex and threadCount into stack-allocated storage so
        // that their symbols point to something reasonable.
        threadIndexSym->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "threadIndex");
        ctx->StoreInst(threadIndex, threadIndexSym->storageInfo);

        threadCountSym->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "threadCount");
        ctx->StoreInst(threadCount, threadCountSym->storageInfo);

        // Copy taskIndex and taskCount into stack-allocated storage so
        // that their symbols point to something reasonable.
        taskIndexSym->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex");
        ctx->StoreInst(taskIndex, taskIndexSym->storageInfo);

        taskCountSym->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount");
        ctx->StoreInst(taskCount, taskCountSym->storageInfo);

        taskIndexSym0->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex0");
        ctx->StoreInst(taskIndex0, taskIndexSym0->storageInfo);
        taskIndexSym1->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex1");
        ctx->StoreInst(taskIndex1, taskIndexSym1->storageInfo);
        taskIndexSym2->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex2");
        ctx->StoreInst(taskIndex2, taskIndexSym2->storageInfo);

        taskCountSym0->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount0");
        ctx->StoreInst(taskCount0, taskCountSym0->storageInfo);
        taskCountSym1->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount1");
        ctx->StoreInst(taskCount1, taskCountSym1->storageInfo);
        taskCountSym2->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount2");
        ctx->StoreInst(taskCount2, taskCountSym2->storageInfo);
    } else {
        // Regular, non-task function or GPU task
        llvm::Function::arg_iterator argIter = function->arg_begin();
        llvm::FunctionType *fType = type->LLVMFunctionType(g->ctx);
        Assert(fType->getFunctionNumParams() >= args.size());
        for (unsigned int i = 0; i < args.size(); ++i, ++argIter) {
            Symbol *argSym = args[i];
            if (argSym == nullptr)
                // anonymous function parameter
                continue;

            argIter->setName(argSym->name.c_str());

            // Allocate stack storage for the parameter and emit code
            // to store the its value there.
            argSym->storageInfo = ctx->AllocaInst(argSym->type, argSym->name.c_str());
            // ISPC export and extern "C" functions have addrspace in the declaration on Xe so
            // we cast addrspace from generic to default in the alloca BB.
            // define dso_local spir_func void @test(%S addrspace(4)* noalias %s)
            // addrspacecast %S addrspace(4)* %s to %S*
            llvm::Value *addrCasted = &*argIter;
#ifdef ISPC_XE_ENABLED
            // Update addrspace of passed argument if needed for Xe target
            if (g->target->isXeTarget()) {
                addrCasted = ctx->XeUpdateAddrSpaceForParam(addrCasted, fType, i, true);
            }
#endif

            ctx->StoreInst(addrCasted, argSym->storageInfo, argSym->type);

            ctx->EmitFunctionParameterDebugInfo(argSym, i);
        }

        // If the number of actual function arguments is equal to the
        // number of declared arguments in decl->functionParams, then we
        // don't have a mask parameter, so set it to be all on.  This
        // happens for example with 'export'ed functions that the app
        // calls, with tasks on GPU and with unmasked functions.
        if (argIter == function->arg_end()) {
            Assert(type->isUnmasked || type->isExported || type->isExternC || type->isExternSYCL ||
                   type->IsISPCExternal() || type->IsISPCKernel());
            ctx->SetFunctionMask(LLVMMaskAllOn);
        } else {
            Assert(type->isUnmasked == false);

            // Otherwise use the mask to set the entry mask value
            argIter->setName("__mask");
            Assert(argIter->getType() == LLVMTypes::MaskType);

            if (ctx->emitXeHardwareMask()) {
                // We should not create explicit predication
                // to avoid EM usage duplication. All stuff
                // will be done by SIMD CF Lowering
                // TODO: temporary workaround that will be changed
                // as part of SPIR-V emitting solution
                ctx->SetFunctionMask(LLVMMaskAllOn);
            } else {
                ctx->SetFunctionMask(&*argIter);
            }

            ++argIter;
            Assert(argIter == function->arg_end());
        }
        if (g->target->isXeTarget() && type->isTask) {
            // Assign threadIndex and threadCount to the result of calling of corresponding builtins.
            // On Xe threadIndex equals to taskIndex and threadCount to taskCount.
            threadIndexSym->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "threadIndex");
            ctx->StoreInst(lXeGetTaskVariableValue(ctx, builtin::__task_index), threadIndexSym->storageInfo);

            threadCountSym->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "threadCount");
            ctx->StoreInst(lXeGetTaskVariableValue(ctx, builtin::__task_count), threadCountSym->storageInfo);

            // Assign taskIndex and taskCount to the result of calling of corresponding builtins.
            taskIndexSym->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex");
            ctx->StoreInst(lXeGetTaskVariableValue(ctx, builtin::__task_index), taskIndexSym->storageInfo);

            taskCountSym->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount");
            ctx->StoreInst(lXeGetTaskVariableValue(ctx, builtin::__task_count), taskCountSym->storageInfo);

            taskIndexSym0->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex0");
            ctx->StoreInst(lXeGetTaskVariableValue(ctx, builtin::__task_index0), taskIndexSym0->storageInfo);
            taskIndexSym1->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex1");
            ctx->StoreInst(lXeGetTaskVariableValue(ctx, builtin::__task_index1), taskIndexSym1->storageInfo);
            taskIndexSym2->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskIndex2");
            ctx->StoreInst(lXeGetTaskVariableValue(ctx, builtin::__task_index2), taskIndexSym2->storageInfo);

            taskCountSym0->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount0");
            ctx->StoreInst(lXeGetTaskVariableValue(ctx, builtin::__task_count0), taskCountSym0->storageInfo);
            taskCountSym1->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount1");
            ctx->StoreInst(lXeGetTaskVariableValue(ctx, builtin::__task_count1), taskCountSym1->storageInfo);
            taskCountSym2->storageInfo = ctx->AllocaInst(LLVMTypes::Int32Type, "taskCount2");
            ctx->StoreInst(lXeGetTaskVariableValue(ctx, builtin::__task_count2), taskCountSym2->storageInfo);
        }
    }

    // Set FTZ/DAZ flags if requested
    ctx->SetFunctionFTZ_DAZFlags();

    // Finally, we can generate code for the function
    if (code != nullptr) {
        ctx->SetDebugPos(code->pos);
        ctx->AddInstrumentationPoint("function entry");

        int costEstimate = EstimateCost(code);
        Debug(code->pos, "Estimated cost for function \"%s\" = %d\n", sym->name.c_str(), costEstimate);

        // If the body of the function is non-trivial, then we wrap the
        // entire thing inside code that tests to see if the mask is all
        // on, all off, or mixed.  If this is a simple function, then this
        // isn't worth the code bloat / overhead.
        bool checkMask =
            (!g->target->isXeTarget() && type->isTask == true) ||
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
            ((function->getAttributes().getFnAttrs().hasAttribute(llvm::Attribute::AlwaysInline) == false) &&
#else
            ((function->getAttributes().getFnAttributes().hasAttribute(llvm::Attribute::AlwaysInline) == false) &&
#endif
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

        // if bblock is non-nullptr, it hasn't been terminated by e.g. a
        // return instruction.  Need to add a return instruction.
        ctx->ReturnInst();
    }
#ifdef ISPC_XE_ENABLED
    if (type->IsISPCKernel()) {
        // Emit metadata for XE kernel

        llvm::LLVMContext &fContext = function->getContext();
        llvm::NamedMDNode *mdKernels = m->module->getOrInsertNamedMetadata("genx.kernels");

        std::string AsmName =
            (m->module->getName() + llvm::Twine('_') + llvm::Twine(mdKernels->getNumOperands()) + llvm::Twine(".asm"))
                .str();

        // Kernel arg kinds
        llvm::Type *i32Type = llvm::Type::getInt32Ty(fContext);
        llvm::SmallVector<llvm::Metadata *, 8> argKinds;
        llvm::SmallVector<llvm::Metadata *, 8> argInOutKinds;
        llvm::SmallVector<llvm::Metadata *, 8> argTypeDescs;

        // In ISPC we need only AK_NORMAL and IK_NORMAL now, in future it can change.
        enum { AK_NORMAL, AK_SAMPLER, AK_SURFACE, AK_VME };
        enum { IK_NORMAL, IK_INPUT, IK_OUTPUT, IK_INPUT_OUTPUT };
        unsigned int offset = 32;
        unsigned int grf_size = g->target->getXeGrfSize();
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

            offset += bytes;
        }

        // TODO: Number of fields is 9 now, and it is a magic number that seems
        // to be not defined anywhere. Consider changing it when possible.
        llvm::SmallVector<llvm::Metadata *, 9> mdArgs(9, nullptr);
        mdArgs[llvm::genx::KernelMDOp::FunctionRef] = llvm::ValueAsMetadata::get(function);
        mdArgs[llvm::genx::KernelMDOp::Name] = llvm::MDString::get(fContext, sym->name);
        mdArgs[llvm::genx::KernelMDOp::ArgKinds] = llvm::MDNode::get(fContext, argKinds);
        mdArgs[llvm::genx::KernelMDOp::SLMSize] = llvm::ValueAsMetadata::get(llvm::ConstantInt::getNullValue(i32Type));
        mdArgs[llvm::genx::KernelMDOp::ArgOffsets] =
            llvm::ValueAsMetadata::get(llvm::ConstantInt::getNullValue(i32Type));
        mdArgs[llvm::genx::KernelMDOp::ArgIOKinds] = llvm::MDNode::get(fContext, argInOutKinds);
        mdArgs[llvm::genx::KernelMDOp::ArgTypeDescs] = llvm::MDNode::get(fContext, argTypeDescs);
        mdArgs[llvm::genx::KernelMDOp::NBarrierCnt] =
            llvm::ValueAsMetadata::get(llvm::ConstantInt::getNullValue(i32Type));
        mdArgs[llvm::genx::KernelMDOp::BarrierCnt] =
            llvm::ValueAsMetadata::get(llvm::ConstantInt::getNullValue(i32Type));

        mdKernels->addOperand(llvm::MDNode::get(fContext, mdArgs));
        // This is needed to run in L0 runtime.
        function->addFnAttr("oclrt", "1");
    }
#endif
}

void Function::GenerateIR() {
    if (sym == nullptr)
        // May be nullptr due to error earlier in compilation
        return;

    llvm::Function *function = sym->function;
    Assert(function != nullptr);

    // But if that function has a definition, we don't want to redefine it.
    if (function->empty() == false) {
        Error(sym->pos, "Ignoring redefinition of function \"%s\".", sym->name.c_str());
        return;
    }

    const FunctionType *type = CastType<FunctionType>(sym->type);
    Assert(type != nullptr);

    if (type->isExternSYCL) {
        Error(sym->pos, "\n\'extern \"SYCL\"\' function \"%s\" cannot be defined in ISPC.", sym->name.c_str());
        return;
    }

    // Figure out a reasonable source file position for the start of the
    // function body.  If possible, get the position of the first actual
    // non-StmtList statment...
    SourcePos firstStmtPos = sym->pos;
    if (code) {
        StmtList *sl = llvm::dyn_cast<StmtList>(code);
        if (sl && sl->stmts.size() > 0 && sl->stmts[0] != nullptr)
            firstStmtPos = sl->stmts[0]->pos;
        else
            firstStmtPos = code->pos;
    }
    // And we can now go ahead and emit the code
    if (g->target->isXeTarget()) {
        // For Xe target we do not emit code for masked version of a function
        // if it is a kernel
        const FunctionType *type = CastType<FunctionType>(sym->type);
        if (!type->IsISPCKernel()) {
            llvm::TimeTraceScope TimeScope("emitCode", llvm::StringRef(sym->name));
            FunctionEmitContext ec(this, sym, function, firstStmtPos);
            emitCode(&ec, function, firstStmtPos);
        }
    } else {
        // In case of multi-target compilation for extern "C" functions which were defined, we want
        // to have a target-specific implementation for each target similar to exported functions.
        // However declarations of extern "C"/"SYCL" functions must be not-mangled and therefore, the calls to such
        // functions must be not-mangled. The trick to support target-specific implementation in such case is to
        // generate definition of target-specific implementation mangled with target ("name_<target>") which would be
        // called from a dispatch function. Since we use not-mangled names in the call, it will be a call to a dispatch
        // function which will resolve to particular implementation. The condition below ensures that in case of
        // multi-target compilation we will emit only one-per-target definition of extern "C" function mangled with
        // <target> suffix.
        if (!((type->isExternC || type->isExternSYCL) && g->mangleFunctionsWithTarget)) {
            llvm::TimeTraceScope TimeScope("emitCode", llvm::StringRef(sym->name));
            FunctionEmitContext ec(this, sym, function, firstStmtPos);
            emitCode(&ec, function, firstStmtPos);
        }
    }

    if (m->errorCount == 0) {
        // If the function is 'export'-qualified, emit a second version of
        // it without a mask parameter and without name mangling so that
        // the application can call it.
        // For 'extern "C"' we emit the version without mask parameter only.
        // For Xe we emit a version without mask parameter only for ISPC kernels and
        // ISPC external functions.
        if (type->isExported || type->isExternC || type->isExternSYCL || type->IsISPCExternal() ||
            type->IsISPCKernel()) {
            llvm::FunctionType *ftype = type->LLVMFunctionType(g->ctx, true);
            llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::ExternalLinkage;
            auto [name_pref, name_suf] = type->GetFunctionMangledName(true);
            std::string functionName = name_pref + sym->name + name_suf;

            llvm::Function *appFunction = llvm::Function::Create(ftype, linkage, functionName.c_str(), m->module);
            appFunction->setDoesNotThrow();
            appFunction->setCallingConv(type->GetCallingConv());

            AddUWTableFuncAttr(appFunction);

            // Xe kernel should have "dllexport" and "CMGenxMain" attribute,
            // otherss have "CMStackCall" attribute
            if (g->target->isXeTarget()) {
                if (type->IsISPCExternal()) {
                    appFunction->addFnAttr("CMStackCall");

                } else if (type->IsISPCKernel()) {
                    appFunction->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
                    appFunction->addFnAttr("CMGenxMain");
                }
            } else {
                // Make application function callable from DLLs.
                if ((g->target_os == TargetOS::windows) && (g->dllExport)) {
                    appFunction->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
                }
            }

            if (function->getFunctionType()->getNumParams() > 0) {
                for (int i = 0; i < function->getFunctionType()->getNumParams() - 1; i++) {
                    if (function->hasParamAttribute(i, llvm::Attribute::NoAlias)) {
                        appFunction->addParamAttr(i, llvm::Attribute::NoAlias);
                    }
                }
            }
            g->target->markFuncWithTargetAttr(appFunction);

            if (appFunction->getName() != functionName) {
                // this was a redefinition for which we already emitted an
                // error, so don't worry about this one...
                appFunction->eraseFromParent();
            } else {
                llvm::TimeTraceScope TimeScope("emitCode", llvm::StringRef(sym->name));
                // And emit the code again
                FunctionEmitContext ec(this, sym, appFunction, firstStmtPos);
                emitCode(&ec, appFunction, firstStmtPos);
                if (m->errorCount == 0) {
                    sym->exportedFunction = appFunction;
                }
            }
        } else {
            // Set linkage for the function
            ispc::StorageClass sc = sym->storageClass;
            bool isInline =
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
                (function->getAttributes().getFnAttrs().hasAttribute(llvm::Attribute::AlwaysInline));
#else
                (function->getAttributes().getFnAttributes().hasAttribute(llvm::Attribute::AlwaysInline));
#endif
            // We create regular functions with ExternalLinkage by default.
            // Fix it to InternalLinkage only if the function is static or inline
            if (sc == SC_STATIC || isInline) {
                function->setLinkage(llvm::GlobalValue::InternalLinkage);
            }

            if (g->target->isXeTarget()) {
                // Mark all internal ISPC functions as a stack call
                function->addFnAttr("CMStackCall");
                // Mark all internal ISPC functions as AlwaysInline to facilitate inlining on GPU
                // if it's not marked as "noinline" explicitly
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
                if (!(function->getAttributes().getFnAttrs().hasAttribute(llvm::Attribute::NoInline) ||
                      function->getAttributes().getFnAttrs().hasAttribute(llvm::Attribute::AlwaysInline)))
#else
                if (!(function->getAttributes().getFnAttributes().hasAttribute(llvm::Attribute::NoInline) ||
                      function->getAttributes().getFnAttributes().hasAttribute(llvm::Attribute::AlwaysInline)))
#endif
                {
                    function->addFnAttr(llvm::Attribute::AlwaysInline);
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// TemplateParam

TemplateParam::TemplateParam(const TemplateTypeParmType *p) : paramType(ParamType::Type), typeParam(p) {
    name = p->GetName();
    pos = p->GetSourcePos();
}

TemplateParam::TemplateParam(Symbol *s) : paramType(ParamType::NonType), nonTypeParam(s) {
    name = s->name;
    pos = s->pos;
}

bool TemplateParam::IsTypeParam() const { return paramType == ParamType::Type; }

bool TemplateParam::IsNonTypeParam() const { return paramType == ParamType::NonType; }

bool TemplateParam::IsEqual(const TemplateParam &other) const {
    if (IsTypeParam()) {
        return Type::Equal(typeParam, other.typeParam);
    } else if (IsNonTypeParam()) {
        return nonTypeParam->name == other.nonTypeParam->name &&
               Type::Equal(nonTypeParam->type, other.nonTypeParam->type);
    }
    return false;
}

std::string TemplateParam::GetName() const { return name; }

const TemplateTypeParmType *TemplateParam::GetTypeParam() const {
    Assert(IsTypeParam());
    return typeParam;
}

Symbol *TemplateParam::GetNonTypeParam() const {
    Assert(IsNonTypeParam());
    return nonTypeParam;
}

SourcePos TemplateParam::GetSourcePos() const { return pos; }

///////////////////////////////////////////////////////////////////////////
// TemplateParms

TemplateParms::TemplateParms() {}

void TemplateParms::Add(const TemplateParam *p) { parms.push_back(p); }

size_t TemplateParms::GetCount() const { return parms.size(); }

const TemplateParam *TemplateParms::operator[](size_t i) const { return parms[i]; }

const TemplateParam *TemplateParms::operator[](size_t i) { return parms[i]; }

bool TemplateParms::IsEqual(const TemplateParms *p) const {
    if (p == nullptr) {
        return false;
    }

    if (GetCount() != p->GetCount()) {
        return false;
    }

    for (size_t i = 0; i < GetCount(); i++) {
        const TemplateParam *other = (*p)[i];
        if (!(parms[i]->IsEqual(*other))) {
            return false;
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////
// TemplateArg

TemplateArg::TemplateArg(const Type *t, SourcePos pos) : argType(ArgType::Type), type(t), pos(pos) {}
TemplateArg::TemplateArg(const Expr *c, SourcePos pos) : argType(ArgType::NonType), expr(c), pos(pos) {}

const Type *TemplateArg::GetAsType() const {
    switch (argType) {
    case ArgType::Type:
        return type;
    case ArgType::NonType:
        return expr->GetType();
    default:
        return nullptr;
    }
}

const Expr *TemplateArg::GetAsExpr() const { return IsNonType() ? expr : nullptr; }

SourcePos TemplateArg::GetPos() const { return pos; }

std::string TemplateArg::GetString() const {
    switch (argType) {
    case ArgType::Type:
        return type->GetString();
    case ArgType::NonType:
        if (const ConstExpr *constExpr = GetAsConstExpr()) {
            return constExpr->GetValuesAsStr(", ");
        }
        return "Missing const expression";
    default:
        return "Unknown ArgType";
    }
}

bool TemplateArg::IsNonType() const { return argType == ArgType::NonType; };

bool TemplateArg::IsType() const { return argType == ArgType::Type; }

bool TemplateArg::operator==(const TemplateArg &other) const {
    if (argType != other.argType)
        return false;
    switch (argType) {
    case ArgType::Type:
        return Type::Equal(type, other.type);
    case ArgType::NonType: {
        const ConstExpr *constExpr = GetAsConstExpr();
        const ConstExpr *otherConstExpr = other.GetAsConstExpr();
        if (constExpr && otherConstExpr) {
            return constExpr->IsEqual(otherConstExpr);
        }
        return false;
    }
    default:
        return false;
    }
    return false;
}

std::string TemplateArg::Mangle() const {
    switch (argType) {
    case ArgType::Type:
        return type->Mangle();
    case ArgType::NonType: {
        if (const ConstExpr *constExpr = GetAsConstExpr()) {
            return GetAsType()->Mangle() + constExpr->GetValuesAsStr("_");
        }
        return "Missing const expression";
    }
    default:
        return "Unknown ArgType";
    }
}

void TemplateArg::SetAsVaryingType() {
    if (IsType() && type->GetVariability() == Variability::Unbound) {
        type = type->GetAsVaryingType();
    }
}

const ConstExpr *TemplateArg::GetAsConstExpr() const {
    if (IsNonType()) {
        const ConstExpr *constExpr = llvm::dyn_cast<ConstExpr>(expr);
        if (!constExpr) {
            const SymbolExpr *symExpr = llvm::dyn_cast<SymbolExpr>(expr);
            if (symExpr->GetBaseSymbol()->constValue) {
                constExpr = llvm::dyn_cast<ConstExpr>(symExpr->GetBaseSymbol()->constValue);
            }
        }
        return constExpr;
    }
    return nullptr;
}

///////////////////////////////////////////////////////////////////////////
// FunctionTemplate

FunctionTemplate::FunctionTemplate(TemplateSymbol *s, Stmt *c) : sym(s), code(c) {
    maskSymbol = m->symbolTable->LookupVariable("__mask");
    Assert(maskSymbol != nullptr);

    const FunctionType *type = GetFunctionType();
    Assert(type != nullptr);

    for (int i = 0; i < type->GetNumParameters(); ++i) {
        const char *paramName = type->GetParameterName(i).c_str();
        Symbol *paramSym = m->symbolTable->LookupVariable(paramName);
        if (paramSym == nullptr) {
            Assert(strncmp(paramName, "__anon_parameter_", 17) == 0);
        }
        args.push_back(paramSym);

        // No initialization of parentFunction, as it's needed only for code generation
        // and hence it doesn't make sense for the template. Instantiations will get it initialized.
    }
}

FunctionTemplate::~FunctionTemplate() {
    for (const auto &inst : instantiations) {
        Function *func = const_cast<Function *>(inst.second->parentFunction);
        if (func) {
            delete func;
        }
        TemplateArgs *templArgs = const_cast<TemplateArgs *>(inst.first);
        if (templArgs) {
            delete templArgs;
        }
    }
}

std::string FunctionTemplate::GetName() const {
    Assert(sym);
    return sym->name;
}

const TemplateParms *FunctionTemplate::GetTemplateParms() const {
    Assert(sym);
    return sym->templateParms;
}

const FunctionType *FunctionTemplate::GetFunctionType() const {
    Assert(sym);
    return sym->type;
}

StorageClass FunctionTemplate::GetStorageClass() {
    Assert(sym);
    return sym->storageClass;
}

void FunctionTemplate::Print() const {
    Indent indent;
    indent.pushSingle();
    Print(indent);
    fflush(stdout);
};

void FunctionTemplate::GenerateIR() const {
    for (const auto &inst : instantiations) {
        Function *func = const_cast<Function *>(inst.second->parentFunction);
        if (func != nullptr) {
            func->GenerateIR();
        } else {
            Error(inst.second->pos, "Template function specialization was declared but never defined.");
        }
    }
}

void FunctionTemplate::Print(Indent &indent) const {
    indent.Print("FunctionTemplate", sym->pos);

    const FunctionType *ftype = GetFunctionType();
    if (ftype) {
        printf("[%s] ", ftype->GetString().c_str());
    }

    printf("\"%s\"\n", GetName().c_str());

    const TemplateParms *typenames = GetTemplateParms();
    int itemsToPrint = typenames->GetCount() + (code ? 1 : 0) + instantiations.size();

    indent.pushList(itemsToPrint);
    if (typenames->GetCount() > 0) {
        for (int i = 0; i < typenames->GetCount(); i++) {
            static constexpr std::size_t BUFSIZE{25};
            char buffer[BUFSIZE];
            snprintf(buffer, BUFSIZE, "template param %d", i);
            indent.setNextLabel(buffer);
            if ((*typenames)[i]) {
                indent.Print((*typenames)[i]->IsTypeParam()
                                 ? "TemplateTypeParmType"
                                 : (*typenames)[i]->GetNonTypeParam()->type->GetString().c_str(),
                             (*typenames)[i]->GetSourcePos());
                printf("\"%s\"\n", (*typenames)[i]->GetName().c_str());
                indent.Done();
            } else {
                indent.Print("<NULL>");
                indent.Done();
            }
        }
    }

    if (code) {
        indent.setNextLabel("body");
        code->Print(indent);
    }

    for (const auto &inst : instantiations) {
        std::string args;
        for (size_t i = 0; i < inst.first->size(); i++) {
            const TemplateArg &arg = (*inst.first)[i];
            args += arg.GetString();
            if (i + 1 < inst.first->size()) {
                args += ", ";
            }
        }
        args = "instantiation <" + args + ">";
        indent.setNextLabel(args);
        inst.second->parentFunction->Print(indent);
    }

    indent.Done();
};

bool FunctionTemplate::IsStdlibSymbol() const {
    if (sym == nullptr) {
        return false;
    }
    if (sym->pos.name != nullptr && !strcmp(sym->pos.name, "stdlib.ispc")) {
        return true;
    }
    return false;
};

Symbol *FunctionTemplate::LookupInstantiation(const TemplateArgs &tArgs) {
    TemplateArgs argsToMatch(tArgs);
    for (const auto &inst : instantiations) {
        if (*(inst.first) == argsToMatch) {
            return inst.second;
        }
    }
    return nullptr;
}

Symbol *FunctionTemplate::AddInstantiation(const TemplateArgs &tArgs, TemplateInstantiationKind kind, bool isInline,
                                           bool isNoinline) {
    const TemplateParms *typenames = GetTemplateParms();
    Assert(typenames);
    TemplateInstantiation templInst(*typenames, tArgs, kind, isInline, isNoinline);

    Symbol *instSym = templInst.InstantiateTemplateSymbol(sym);
    Symbol *instMaskSym = templInst.InstantiateSymbol(maskSymbol);
    std::vector<Symbol *> instArgs;
    for (auto arg : args) {
        instArgs.push_back(templInst.InstantiateSymbol(arg));
    }

    Stmt *instCode = code->Instantiate(templInst);
    Function *inst = new Function(instSym, instCode, instMaskSym, instArgs);

    templInst.SetFunction(inst);

    TemplateArgs *templArgs = new TemplateArgs(tArgs);
    instantiations.push_back(std::make_pair(templArgs, instSym));

    return instSym;
}

Symbol *FunctionTemplate::AddSpecialization(const FunctionType *ftype, const TemplateArgs &tArgs, bool isInline,
                                            bool isNoInline, SourcePos pos) {
    const TemplateParms *typenames = GetTemplateParms();
    Assert(typenames);
    TemplateInstantiation templInst(*typenames, tArgs, TemplateInstantiationKind::Specialization, isInline, isNoInline);

    // Create a function symbol
    Symbol *instSym = templInst.InstantiateTemplateSymbol(sym);
    // Inherit unmasked specifier and storageClass from the basic template.
    const FunctionType *instType = CastType<FunctionType>(sym->type);
    bool instUnmasked = instType ? instType->isUnmasked : false;
    instSym->type = instUnmasked ? ftype->GetAsUnmaskedType() : ftype->GetAsNonUnmaskedType();
    instSym->pos = pos;
    instSym->storageClass = sym->storageClass;

    TemplateArgs *templArgs = new TemplateArgs(tArgs);

    // Check if we have previously declared specialization and we are about to define it.
    Symbol *funcSym = LookupInstantiation(tArgs);
    if (funcSym != nullptr) {
        delete templArgs;
        return funcSym;
    } else {
        instantiations.push_back(std::make_pair(templArgs, instSym));
    }
    return instSym;
}

///////////////////////////////////////////////////////////////////////////
// TemplateInstantiation

TemplateInstantiation::TemplateInstantiation(const TemplateParms &typeParms, const TemplateArgs &tArgs,
                                             TemplateInstantiationKind k, bool ii, bool ini)
    : functionSym(nullptr), kind(k), isInline(ii), isNoInline(ini) {
    Assert(tArgs.size() <= typeParms.GetCount());
    // Create a mapping from the template parameters to the arguments.
    // Note we do that for all specified templates arguments, which number may be less than a number of template
    // parameters. In this case the rest of template parameters will be deduced later during template argumnet
    // deduction.
    for (int i = 0; i < tArgs.size(); i++) {
        std::string name = typeParms[i]->GetName();
        const TemplateArg *arg = new TemplateArg(tArgs[i]);
        argsMap[name] = arg;
        templateArgs.push_back(tArgs[i]);
    }
}

void TemplateInstantiation::AddArgument(std::string paramName, TemplateArg arg) {
    const TemplateArg *argPtr = new TemplateArg(arg);
    argsMap[paramName] = argPtr;
}

const Type *TemplateInstantiation::InstantiateType(const std::string &name) {
    auto t = argsMap.find(name);
    if (t == argsMap.end()) {
        return nullptr;
    }

    return t->second->GetAsType();
}

Symbol *TemplateInstantiation::InstantiateSymbol(Symbol *sym) {
    if (sym == nullptr) {
        return nullptr;
    }

    // A note about about global symbols.
    // In the current state of symbol table there's no clear way to differentiate between global and local symbols.
    // There's "parentFunction" field, but it's empty for some local symbols and paramters, which prevents using it
    // for the purpose of differentiation.
    // There's another possible way to differentiate - "storageInfo" tends to be set only for global symbols, but again
    // it's inderent and unreliable way to detect what needs to be encoded explicitly.
    // So we copy all symbols - global and local, while we need not avoid copying globals.
    // TODO: develop a reliable mechanism to detect global symbols and do not copy them.

    auto t = symMap.find(sym);
    if (t != symMap.end()) {
        return t->second;
    }

    const Type *instType = sym->type->ResolveDependenceForTopType(*this);
    Symbol *instSym = new Symbol(sym->name, sym->pos, instType, sym->storageClass);
    // Update constValue for non-type template parameters
    if (argsMap.find(sym->name) != argsMap.end()) {
        const TemplateArg *arg = argsMap[sym->name];
        Assert(arg != nullptr);
        const ConstExpr *ce = arg->GetAsConstExpr();
        if (ce != nullptr) {
            // Do a little type cast to the actual template parameter type here and optimize it
            Expr *castExpr = new TypeCastExpr(sym->type, const_cast<ConstExpr *>(ce), sym->pos);
            castExpr = Optimize(castExpr);
            ce = llvm::dyn_cast<ConstExpr>(castExpr);
        }
        instSym->constValue = ce ? ce->Instantiate(*this) : nullptr;
    } else {
        instSym->constValue = sym->constValue ? sym->constValue->Instantiate(*this) : nullptr;
    }

    instSym->varyingCFDepth = sym->varyingCFDepth;
    instSym->parentFunction = nullptr;
    instSym->storageInfo = sym->storageInfo;

    symMap.emplace(std::make_pair(sym, instSym));
    return instSym;
}

Symbol *TemplateInstantiation::InstantiateTemplateSymbol(TemplateSymbol *sym) {
    // The function is assumed to be called once per instantiation and
    // only for the tempalte that is being instantiated.
    Assert(sym && functionSym == nullptr);

    // Instantiate the function type
    const Type *instType = sym->type->ResolveDependenceForTopType(*this);

    // Create a function symbol
    Symbol *instSym = new Symbol(sym->name, sym->pos, instType, sym->storageClass);
    functionSym = instSym;

    // Create llvm::Function and attach to the symbol, so the symbol is complete and ready for use.
    llvm::Function *llvmFunc = createLLVMFunction(instSym);
    instSym->function = llvmFunc;
    return instSym;
}

// After the instance of the template function is created, the symbols should point to the parent function.
void TemplateInstantiation::SetFunction(Function *func) {
    for (auto &symPair : symMap) {
        Symbol *sym = symPair.second;
        sym->parentFunction = func;
    }
    functionSym->parentFunction = func;
}

// For regular functions, llvm::Function is create when declaration is met in the program to ensure that
// the function symbol is represented llvm::Module as declaration. So all the work is done in ispc::Module.
// For function templates we need llvm::Function when instantiation is created, so we do it here.
// TODO: change the design to unify llvm::Function creation for both regular functions and instantiations of
// function templates.
llvm::Function *TemplateInstantiation::createLLVMFunction(Symbol *functionSym) {
    Assert(functionSym && functionSym->type && CastType<FunctionType>(functionSym->type));
    const FunctionType *functionType = CastType<FunctionType>(functionSym->type);

    // Get the LLVM FunctionType
    llvm::FunctionType *llvmFunctionType = functionType->LLVMFunctionType(g->ctx, false /*disableMask*/);
    if (llvmFunctionType == nullptr) {
        return nullptr;
    }

    // Mangling
    auto [name_pref, name_suf] = functionType->GetFunctionMangledName(false, &templateArgs);
    std::string functionName = name_pref + functionSym->name + name_suf;

    llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::ExternalLinkage;
    if (functionSym->storageClass == SC_STATIC || isInline) {
        linkage = llvm::GlobalValue::InternalLinkage;
    } else {
        // If the linkage is not internal, apply the Clang linkage rules for templates.
        switch (kind) {
        // Function can be defined multiple times across different translation units without causing conflicts.
        // The linker will choose a definition for the function based on its default behavior.
        case TemplateInstantiationKind::Explicit:
            linkage = llvm::GlobalValue::WeakODRLinkage;
            break;
        // The function is only allowed to be defined once across all translation units, but it can be discarded if
        // unused. If multiple definitions of the function are present across different translation units, the linker
        // will keep only one of them, discarding the rest.
        case TemplateInstantiationKind::Implicit:
            linkage = llvm::GlobalValue::LinkOnceODRLinkage;
            break;
        case TemplateInstantiationKind::Specialization:
            linkage = llvm::GlobalValue::ExternalLinkage;
            break;
        default:
            break;
        }
    }
    // And create the llvm::Function
    llvm::Function *function = llvm::Function::Create(llvmFunctionType, linkage, functionName.c_str(), m->module);

    // Set function attributes: we never throw exceptions
    function->setDoesNotThrow();

    function->setCallingConv(functionType->GetCallingConv());
    g->target->markFuncWithTargetAttr(function);

    if (isInline) {
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    if (isNoInline) {
        function->addFnAttr(llvm::Attribute::NoInline);
    }

    AddUWTableFuncAttr(function);

    // Add NoAlias attribute to function arguments if needed.
    int nArgs = functionType->GetNumParameters();
    for (int i = 0; i < nArgs; ++i) {
        const Type *argType = functionType->GetParameterType(i);

        // ISPC assumes that no pointers alias.  (It should be possible to
        // specify when this is not the case, but this should be the
        // default.)  Set parameter attributes accordingly.  (Only for
        // uniform pointers, since varying pointers are int vectors...)
        if (!functionType->isTask && !functionType->isExternSYCL &&
            ((CastType<PointerType>(argType) != nullptr && argType->IsUniformType() &&
              // Exclude SOA argument because it is a pair {struct *, int}
              // instead of pointer
              !CastType<PointerType>(argType)->IsSlice()) ||

             CastType<ReferenceType>(argType) != nullptr)) {

            function->addParamAttr(i, llvm::Attribute::NoAlias);
        }
    }

    // If llvm gave us back a Function * with a different name than the one
    // we asked for, then there's already a function with that same
    // (mangled) name in the llvm::Module.  In that case, erase the one we
    // tried to add and just work with the one it already had.
    if (function->getName() != functionName) {
        function->eraseFromParent();
        function = m->module->getFunction(functionName);
    }

    return function;
}
