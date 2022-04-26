/*
  Copyright (c) 2010-2022, Intel Corporation
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

/** @file opt.cpp
    @brief Implementations of various ispc optimization passes that operate
           on the LLVM IR.
*/

#include "opt.h"
#include "ctx.h"
#include "llvmutil.h"
#include "module.h"
#include "sym.h"
#include "util.h"

#include <map>
#include <regex>
#include <set>
#include <stdio.h>

#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/IR/Verifier.h>
#include <llvm/InitializePasses.h>
#include <llvm/Pass.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/FunctionAttrs.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/InstSimplifyPass.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#ifdef ISPC_HOST_IS_LINUX
#include <alloca.h>
#elif defined(ISPC_HOST_IS_WINDOWS)
#include <malloc.h>
#ifndef __MINGW32__
#define alloca _alloca
#endif
#endif // ISPC_HOST_IS_WINDOWS

#ifndef PRId64
#define PRId64 "lld"
#endif
#ifndef PRIu64
#define PRIu64 "llu"
#endif

#ifndef ISPC_NO_DUMPS
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Regex.h>
#endif
#ifdef ISPC_XE_ENABLED
#include "xe/GlobalsLocalization.h"
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include <llvm/GenXIntrinsics/GenXIntrOpts.h>
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#include <llvm/GenXIntrinsics/GenXSPIRVWriterAdaptor.h>
// Used for Xe gather coalescing
#include <llvm/Transforms/Utils/Local.h>

// Constant in number of bytes.
enum { BYTE = 1, WORD = 2, DWORD = 4, QWORD = 8, OWORD = 16, GRF = 32 };
#endif

using namespace ispc;

static llvm::Pass *CreateIntrinsicsOptPass();
static llvm::Pass *CreateInstructionSimplifyPass();
static llvm::Pass *CreatePeepholePass();

static llvm::Pass *CreateImproveMemoryOpsPass();
static llvm::Pass *CreateGatherCoalescePass();
static llvm::Pass *CreateReplacePseudoMemoryOpsPass();

static llvm::Pass *CreateIsCompileTimeConstantPass(bool isLastTry);
static llvm::Pass *CreateMakeInternalFuncsStaticPass();

#ifndef ISPC_NO_DUMPS
static llvm::Pass *CreateDebugPass(char *output);
static llvm::Pass *CreateDebugPassFile(int number, llvm::StringRef name);
#endif

static llvm::Pass *CreateReplaceStdlibShiftPass();

#ifdef ISPC_XE_ENABLED
static llvm::Pass *CreateXeGatherCoalescingPass();
static llvm::Pass *CreateReplaceLLVMIntrinsics();
static llvm::Pass *CreateDemotePHIs();
static llvm::Pass *CreateCheckUnsupportedInsts();
static llvm::Pass *CreateMangleOpenCLBuiltins();
#endif

#ifndef ISPC_NO_DUMPS
#define DEBUG_START_PASS(NAME)                                                                                         \
    if (g->debugPrint &&                                                                                               \
        (getenv("FUNC") == NULL || (getenv("FUNC") != NULL && !strncmp(bb.getParent()->getName().str().c_str(),        \
                                                                       getenv("FUNC"), strlen(getenv("FUNC")))))) {    \
        fprintf(stderr, "Start of " NAME "\n");                                                                        \
        fprintf(stderr, "---------------\n");                                                                          \
        bb.dump();                                                                                                     \
        fprintf(stderr, "---------------\n\n");                                                                        \
    } else /* eat semicolon */

#define DEBUG_END_PASS(NAME)                                                                                           \
    if (g->debugPrint &&                                                                                               \
        (getenv("FUNC") == NULL || (getenv("FUNC") != NULL && !strncmp(bb.getParent()->getName().str().c_str(),        \
                                                                       getenv("FUNC"), strlen(getenv("FUNC")))))) {    \
        fprintf(stderr, "End of " NAME " %s\n", modifiedAny ? "** CHANGES **" : "");                                   \
        fprintf(stderr, "---------------\n");                                                                          \
        bb.dump();                                                                                                     \
        fprintf(stderr, "---------------\n\n");                                                                        \
    } else /* eat semicolon */
#else
#define DEBUG_START_PASS(NAME)
#define DEBUG_END_PASS(NAME)
#endif

///////////////////////////////////////////////////////////////////////////

/** This utility routine copies the metadata (if any) attached to the
    'from' instruction in the IR to the 'to' instruction.

    For flexibility, this function takes an llvm::Value rather than an
    llvm::Instruction for the 'to' parameter; at some places in the code
    below, we sometimes use a llvm::Value to start out storing a value and
    then later store instructions.  If a llvm::Value is passed to this, the
    routine just returns without doing anything; if it is in fact an
    LLVM::Instruction, then the metadata can be copied to it.
 */
static void lCopyMetadata(llvm::Value *vto, const llvm::Instruction *from) {
    llvm::Instruction *to = llvm::dyn_cast<llvm::Instruction>(vto);
    if (!to)
        return;

    llvm::SmallVector<std::pair<unsigned int, llvm::MDNode *>, 8> metadata;

    from->getAllMetadata(metadata);
    for (unsigned int i = 0; i < metadata.size(); ++i)
        to->setMetadata(metadata[i].first, metadata[i].second);
}

/** We have a protocol with the front-end LLVM IR code generation process
    that allows us to encode the source file position that corresponds with
    instructions.  (For example, this allows us to issue performance
    warnings related to things like scatter and gather after optimization
    has been performed, so that we aren't warning about scatters and
    gathers that have been improved to stores and loads by optimization
    passes.)  Note that this is slightly redundant with the source file
    position encoding generated for debugging symbols, though we don't
    always generate debugging information but we do always generate this
    position data.

    This function finds the SourcePos that the metadata in the instruction
    (if present) corresponds to.  See the implementation of
    FunctionEmitContext::addGSMetadata(), which encodes the source position during
    code generation.

    @param inst   Instruction to try to find the source position of
    @param pos    Output variable in which to store the position
    @returns      True if source file position metadata was present and *pos
                  has been set.  False otherwise.
*/
static bool lGetSourcePosFromMetadata(const llvm::Instruction *inst, SourcePos *pos) {
    llvm::MDNode *filename = inst->getMetadata("filename");
    llvm::MDNode *first_line = inst->getMetadata("first_line");
    llvm::MDNode *first_column = inst->getMetadata("first_column");
    llvm::MDNode *last_line = inst->getMetadata("last_line");
    llvm::MDNode *last_column = inst->getMetadata("last_column");

    if (!filename || !first_line || !first_column || !last_line || !last_column)
        return false;

    // All of these asserts are things that FunctionEmitContext::addGSMetadata() is
    // expected to have done in its operation
    llvm::MDString *str = llvm::dyn_cast<llvm::MDString>(filename->getOperand(0));
    Assert(str);
    llvm::ConstantInt *first_lnum =

        llvm::mdconst::extract<llvm::ConstantInt>(first_line->getOperand(0));
    Assert(first_lnum);

    llvm::ConstantInt *first_colnum =

        llvm::mdconst::extract<llvm::ConstantInt>(first_column->getOperand(0));
    Assert(first_column);

    llvm::ConstantInt *last_lnum =

        llvm::mdconst::extract<llvm::ConstantInt>(last_line->getOperand(0));
    Assert(last_lnum);

    llvm::ConstantInt *last_colnum = llvm::mdconst::extract<llvm::ConstantInt>(last_column->getOperand(0));
    Assert(last_column);

    *pos = SourcePos(str->getString().data(), (int)first_lnum->getZExtValue(), (int)first_colnum->getZExtValue(),
                     (int)last_lnum->getZExtValue(), (int)last_colnum->getZExtValue());
    return true;
}

static llvm::Instruction *lCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, const llvm::Twine &name,
                                    llvm::Instruction *insertBefore = NULL) {
    llvm::Value *args[2] = {arg0, arg1};
    llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[2]);
    return llvm::CallInst::Create(func, newArgArray, name, insertBefore);
}

static llvm::Instruction *lCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, llvm::Value *arg2,
                                    const llvm::Twine &name, llvm::Instruction *insertBefore = NULL) {
    llvm::Value *args[3] = {arg0, arg1, arg2};
    llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[3]);
    return llvm::CallInst::Create(func, newArgArray, name, insertBefore);
}

static llvm::Instruction *lCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, llvm::Value *arg2,
                                    llvm::Value *arg3, const llvm::Twine &name,
                                    llvm::Instruction *insertBefore = NULL) {
    llvm::Value *args[4] = {arg0, arg1, arg2, arg3};
    llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[4]);
    return llvm::CallInst::Create(func, newArgArray, name, insertBefore);
}

static llvm::Instruction *lCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, llvm::Value *arg2,
                                    llvm::Value *arg3, llvm::Value *arg4, const llvm::Twine &name,
                                    llvm::Instruction *insertBefore = NULL) {
    llvm::Value *args[5] = {arg0, arg1, arg2, arg3, arg4};
    llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[5]);
    return llvm::CallInst::Create(func, newArgArray, name, insertBefore);
}

static llvm::Instruction *lCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, llvm::Value *arg2,
                                    llvm::Value *arg3, llvm::Value *arg4, llvm::Value *arg5, const llvm::Twine &name,
                                    llvm::Instruction *insertBefore = NULL) {
    llvm::Value *args[6] = {arg0, arg1, arg2, arg3, arg4, arg5};
    llvm::ArrayRef<llvm::Value *> newArgArray(&args[0], &args[6]);
    return llvm::CallInst::Create(func, newArgArray, name, insertBefore);
}

static llvm::Instruction *lGEPInst(llvm::Value *ptr, llvm::Value *offset, const char *name,
                                   llvm::Instruction *insertBefore) {
    llvm::Value *index[1] = {offset};
    llvm::ArrayRef<llvm::Value *> arrayRef(&index[0], &index[1]);

    return llvm::GetElementPtrInst::Create(PTYPE(ptr), ptr, arrayRef, name, insertBefore);
}

/** Given a vector of constant values (int, float, or bool) representing an
    execution mask, convert it to a bitvector where the 0th bit corresponds
    to the first vector value and so forth.
*/
static uint64_t lConstElementsToMask(const llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> &elements) {
    Assert(elements.size() <= 64);

    uint64_t mask = 0;
    uint64_t undefSetMask = 0;
    llvm::APInt intMaskValue;
    for (unsigned int i = 0; i < elements.size(); ++i) {
        // SSE has the "interesting" approach of encoding blending
        // masks as <n x float>.
        if (llvm::ConstantFP *cf = llvm::dyn_cast<llvm::ConstantFP>(elements[i])) {
            llvm::APFloat apf = cf->getValueAPF();
            intMaskValue = apf.bitcastToAPInt();
        } else if (llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(elements[i])) {
            // Otherwise get it as an int
            intMaskValue = ci->getValue();
        } else {
            // We create a separate 'undef mask' with all undef bits set.
            // This mask will have no bits set if there are no 'undef' elements.
            llvm::UndefValue *uv = llvm::dyn_cast<llvm::UndefValue>(elements[i]);
            Assert(uv != NULL); // vs return -1 if NULL?
            undefSetMask |= (1ull << i);
            continue;
        }
        // Is the high-bit set?  If so, OR in the appropriate bit in
        // the result mask
        if (intMaskValue.countLeadingOnes() > 0)
            mask |= (1ull << i);
    }

    // if no bits are set in mask, do not need to consider undefs. It's
    // always 'all_off'.
    // If any bits are set in mask, assume' undef' bits as as '1'. This ensures
    // cases with only '1's and 'undef's will be considered as 'all_on'
    if (mask != 0)
        mask |= undefSetMask;

    return mask;
}

/** Given an llvm::Value represinting a vector mask, see if the value is a
    constant.  If so, return true and set *bits to be the integer mask
    found by taking the high bits of the mask values in turn and
    concatenating them into a single integer.  In other words, given the
    4-wide mask: < 0xffffffff, 0, 0, 0xffffffff >, we have 0b1001 = 9.
 */
static bool lGetMask(llvm::Value *factor, uint64_t *mask) {
    llvm::ConstantDataVector *cdv = llvm::dyn_cast<llvm::ConstantDataVector>(factor);
    if (cdv != NULL) {
        llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> elements;
        for (int i = 0; i < (int)cdv->getNumElements(); ++i)
            elements.push_back(cdv->getElementAsConstant(i));
        *mask = lConstElementsToMask(elements);
        return true;
    }

    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(factor);
    if (cv != NULL) {
        llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> elements;
        for (int i = 0; i < (int)cv->getNumOperands(); ++i) {
            llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(cv->getOperand(i));
            if (c == NULL)
                return false;
            if (llvm::isa<llvm::ConstantExpr>(cv->getOperand(i)))
                return false; // We can not handle constant expressions here
            elements.push_back(c);
        }
        *mask = lConstElementsToMask(elements);
        return true;
    } else if (llvm::isa<llvm::ConstantAggregateZero>(factor)) {
        *mask = 0;
        return true;
    } else {
#if 0
        llvm::ConstantExpr *ce = llvm::dyn_cast<llvm::ConstantExpr>(factor);
        if (ce != NULL) {
            llvm::TargetMachine *targetMachine = g->target->GetTargetMachine();
            const llvm::TargetData *td = targetMachine->getTargetData();
            llvm::Constant *c = llvm::ConstantFoldConstantExpression(ce, td);
            c->dump();
            factor = c;
        }
        // else we should be able to handle it above...
        Assert(!llvm::isa<llvm::Constant>(factor));
#endif
        return false;
    }
}

enum class MaskStatus { all_on, all_off, mixed, unknown };

/** Determines if the given mask value is all on, all off, mixed, or
    unknown at compile time.
*/
static MaskStatus lGetMaskStatus(llvm::Value *mask, int vecWidth = -1) {
    uint64_t bits;
    if (lGetMask(mask, &bits) == false)
        return MaskStatus::unknown;

    if (bits == 0)
        return MaskStatus::all_off;

    if (vecWidth == -1)
        vecWidth = g->target->getVectorWidth();
    Assert(vecWidth <= 64);

    for (int i = 0; i < vecWidth; ++i) {
        if ((bits & (1ull << i)) == 0)
            return MaskStatus::mixed;
    }
    return MaskStatus::all_on;
}

///////////////////////////////////////////////////////////////////////////
// This is a wrap over class llvm::PassManager. This duplicates PassManager function run()
//   and change PassManager function add by adding some checks and debug passes.
//   This wrap can control:
//   - If we want to switch off optimization with given number.
//   - If we want to dump LLVM IR after optimization with given number.
//   - If we want to generate LLVM IR debug for gdb after optimization with given number.
class DebugPassManager {
  public:
    DebugPassManager() : number(0) {}
    void add(llvm::Pass *P, int stage);
    bool run(llvm::Module &M) { return PM.run(M); }
    llvm::legacy::PassManager &getPM() { return PM; }

  private:
    llvm::legacy::PassManager PM;
    int number;
};

void DebugPassManager::add(llvm::Pass *P, int stage = -1) {
    // taking number of optimization
    if (stage == -1) {
        number++;
    } else {
        number = stage;
    }
    if (g->off_stages.find(number) == g->off_stages.end()) {
        // adding optimization (not switched off)
        PM.add(P);
#ifndef ISPC_NO_DUMPS
        if (g->debug_stages.find(number) != g->debug_stages.end()) {
            // adding dump of LLVM IR after optimization
            if (g->dumpFile) {
                PM.add(CreateDebugPassFile(number, P->getPassName()));
            } else {
                char buf[100];
                snprintf(buf, sizeof(buf), "\n\n*****LLVM IR after phase %d: %s*****\n\n", number,
                         P->getPassName().data());
                PM.add(CreateDebugPass(buf));
            }
        }
#endif
    }
}
///////////////////////////////////////////////////////////////////////////

void ispc::Optimize(llvm::Module *module, int optLevel) {
#ifndef ISPC_NO_DUMPS
    if (g->debugPrint) {
        printf("*** Code going into optimization ***\n");
        module->dump();
    }
#endif
    DebugPassManager optPM;

    if (g->enableLLVMIntrinsics) {
        // Required for matrix intrinsics. This needs to happen before VerifierPass.
        // TODO : Limit pass to only when llvm.matrix.* intrinsics are used.
        optPM.add(llvm::createLowerMatrixIntrinsicsPass()); // llvm.matrix
    }
    optPM.add(llvm::createVerifierPass(), 0);

    optPM.add(new llvm::TargetLibraryInfoWrapperPass(llvm::Triple(module->getTargetTriple())));
    if (!g->target->isXeTarget()) {
        llvm::TargetMachine *targetMachine = g->target->GetTargetMachine();
        optPM.getPM().add(createTargetTransformInfoWrapperPass(targetMachine->getTargetIRAnalysis()));
    }
    optPM.add(llvm::createIndVarSimplifyPass());

#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
    llvm::SimplifyCFGOptions simplifyCFGopt;
    simplifyCFGopt.HoistCommonInsts = true;
#endif
    if (optLevel == 0) {
        // This is more or less the minimum set of optimizations that we
        // need to do to generate code that will actually run.  (We can't
        // run absolutely no optimizations, since the front-end needs us to
        // take the various __pseudo_* functions it has emitted and turn
        // them into something that can actually execute.
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            // Global DCE is required for ISPCSimdCFLoweringPass
            optPM.add(llvm::createGlobalDCEPass());
            // FIXME: temporary solution
            optPM.add(llvm::createBreakCriticalEdgesPass());
            optPM.add(CreateDemotePHIs());
            optPM.add(llvm::createISPCSimdCFLoweringPass());
            // FIXME: temporary solution
            optPM.add(llvm::createPromoteMemoryToRegisterPass());
        }
#endif
        optPM.add(CreateImproveMemoryOpsPass(), 100);

        if (g->opt.disableHandlePseudoMemoryOps == false)
            optPM.add(CreateReplacePseudoMemoryOpsPass());

        optPM.add(CreateIntrinsicsOptPass(), 102);
        optPM.add(CreateIsCompileTimeConstantPass(true));
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(CreateMakeInternalFuncsStaticPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.add(llvm::createPromoteMemoryToRegisterPass());
            optPM.add(llvm::createGlobalsLocalizationPass());
            // Remove dead globals after localization
            optPM.add(llvm::createGlobalDCEPass());
            // This pass is needed for correct prints work
            optPM.add(llvm::createSROAPass());
            optPM.add(CreateReplaceLLVMIntrinsics());
            optPM.add(CreateCheckUnsupportedInsts());
            optPM.add(CreateMangleOpenCLBuiltins());
            // This pass is required to prepare LLVM IR for open source SPIR-V translator
            optPM.add(
                llvm::createGenXSPIRVWriterAdaptorPass(true /*RewriteTypes*/, false /*RewriteSingleElementVectors*/));
        }
#endif
        optPM.add(llvm::createGlobalDCEPass());
    } else {
        llvm::PassRegistry *registry = llvm::PassRegistry::getPassRegistry();
        llvm::initializeCore(*registry);
        llvm::initializeScalarOpts(*registry);
        llvm::initializeIPO(*registry);
        llvm::initializeAnalysis(*registry);
        llvm::initializeTransformUtils(*registry);
        llvm::initializeInstCombine(*registry);
        llvm::initializeInstrumentation(*registry);
        llvm::initializeTarget(*registry);

        optPM.add(llvm::createGlobalDCEPass(), 184);

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            // FIXME: temporary solution
            optPM.add(llvm::createBreakCriticalEdgesPass());
            optPM.add(CreateDemotePHIs());
            optPM.add(llvm::createISPCSimdCFLoweringPass());
            // FIXME: temporary solution
            optPM.add(llvm::createPromoteMemoryToRegisterPass());
        }
#endif
        // Setup to use LLVM default AliasAnalysis
        // Ideally, we want call:
        //    llvm::PassManagerBuilder pm_Builder;
        //    pm_Builder.OptLevel = optLevel;
        //    pm_Builder.addInitialAliasAnalysisPasses(optPM);
        // but the addInitialAliasAnalysisPasses() is a private function
        // so we explicitly enable them here.
        // Need to keep sync with future LLVM change
        // An alternative is to call populateFunctionPassManager()
        optPM.add(llvm::createTypeBasedAAWrapperPass(), 190);
        optPM.add(llvm::createBasicAAWrapperPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif

        optPM.add(llvm::createSROAPass());

        optPM.add(llvm::createEarlyCSEPass());
        optPM.add(llvm::createLowerExpectIntrinsicPass());

        // Early optimizations to try to reduce the total amount of code to
        // work with if we can
        optPM.add(llvm::createReassociatePass(), 200);
        optPM.add(llvm::createInstSimplifyLegacyPass());
        optPM.add(llvm::createDeadCodeEliminationPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif

        optPM.add(llvm::createPromoteMemoryToRegisterPass());
        optPM.add(llvm::createAggressiveDCEPass());

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.add(llvm::createInstructionCombiningPass(), 210);
            optPM.add(CreateImproveMemoryOpsPass());
        }
        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.add(CreateIntrinsicsOptPass(), 215);
            optPM.add(CreateInstructionSimplifyPass());
        }
        optPM.add(llvm::createDeadCodeEliminationPass(), 220);

        // On to more serious optimizations
        optPM.add(llvm::createSROAPass());
        optPM.add(llvm::createInstructionCombiningPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif
        optPM.add(llvm::createPromoteMemoryToRegisterPass());
        optPM.add(llvm::createGlobalOptimizerPass());
        optPM.add(llvm::createReassociatePass());
        // IPConstProp will not be supported by LLVM moving forward.
        // Switching to IPSCCP which is its recommended functional equivalent.
        // TODO : Make IPSCCP the default after ISPC 1.14 release.
#if ISPC_LLVM_VERSION < ISPC_LLVM_12_0
        optPM.add(llvm::createIPConstantPropagationPass());
#else
        optPM.add(llvm::createIPSCCPPass());
#endif

        optPM.add(CreateReplaceStdlibShiftPass(), 229);

        optPM.add(llvm::createDeadArgEliminationPass(), 230);
        optPM.add(llvm::createInstructionCombiningPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif
        optPM.add(llvm::createPruneEHPass());
        optPM.add(llvm::createPostOrderFunctionAttrsLegacyPass());
        optPM.add(llvm::createReversePostOrderFunctionAttrsPass());

        // Next inline pass will remove functions, saved by __keep_funcs_live
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(llvm::createInstSimplifyLegacyPass());
        optPM.add(llvm::createDeadCodeEliminationPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif

        optPM.add(llvm::createArgumentPromotionPass());

        optPM.add(llvm::createAggressiveDCEPass());
        optPM.add(llvm::createInstructionCombiningPass(), 241);
        optPM.add(llvm::createJumpThreadingPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif

        optPM.add(llvm::createSROAPass());

        optPM.add(llvm::createInstructionCombiningPass());
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            // Inline
            optPM.add(llvm::createCorrelatedValuePropagationPass());
            optPM.add(llvm::createInstructionCombiningPass());
            optPM.add(llvm::createGlobalDCEPass());
            optPM.add(llvm::createInstructionCombiningPass());
            optPM.add(llvm::createEarlyCSEPass());
            optPM.add(llvm::createDeadCodeEliminationPass());
            optPM.add(llvm::createGlobalsLocalizationPass());
            // remove dead globals after localization
            optPM.add(llvm::createGlobalDCEPass());
        }
#endif
        optPM.add(llvm::createTailCallEliminationPass());

        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.add(CreateIntrinsicsOptPass(), 250);
            optPM.add(CreateInstructionSimplifyPass());
        }

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.add(llvm::createInstructionCombiningPass(), 255);
#ifdef ISPC_XE_ENABLED
            if (g->target->isXeTarget() && !g->opt.disableXeGatherCoalescing)
                optPM.add(CreateXeGatherCoalescingPass());
#endif
            optPM.add(CreateImproveMemoryOpsPass());

            if (g->opt.disableCoalescing == false) {
                // It is important to run this here to make it easier to
                // finding matching gathers we can coalesce..
                optPM.add(llvm::createEarlyCSEPass(), 260);
                optPM.add(CreateGatherCoalescePass());
            }
        }

        optPM.add(llvm::createFunctionInliningPass(), 265);
        optPM.add(llvm::createInstSimplifyLegacyPass());
        optPM.add(CreateIntrinsicsOptPass());
        optPM.add(CreateInstructionSimplifyPass());

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.add(llvm::createInstructionCombiningPass(), 270);
            optPM.add(CreateImproveMemoryOpsPass());
        }

        optPM.add(llvm::createIPSCCPPass(), 275);
        optPM.add(llvm::createDeadArgEliminationPass());
        optPM.add(llvm::createAggressiveDCEPass());
        optPM.add(llvm::createInstructionCombiningPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif

        if (g->opt.disableHandlePseudoMemoryOps == false) {
            optPM.add(CreateReplacePseudoMemoryOpsPass(), 280);
        }
        optPM.add(CreateIntrinsicsOptPass(), 281);
        optPM.add(CreateInstructionSimplifyPass());

        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(llvm::createArgumentPromotionPass());

        optPM.add(llvm::createSROAPass());

        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(CreateInstructionSimplifyPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif
        optPM.add(llvm::createReassociatePass());
        optPM.add(llvm::createLoopRotatePass());
        optPM.add(llvm::createLICMPass());
        optPM.add(llvm::createLoopUnswitchPass(false));
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(CreateInstructionSimplifyPass());
        optPM.add(llvm::createIndVarSimplifyPass());
        // Currently CM does not support memset/memcpy
        // so this pass is temporary disabled for Xe.
        if (!g->target->isXeTarget()) {
            optPM.add(llvm::createLoopIdiomPass());
        }
        optPM.add(llvm::createLoopDeletionPass());
        if (g->opt.unrollLoops) {
            optPM.add(llvm::createLoopUnrollPass(), 300);
        }
        optPM.add(llvm::createGVNPass(), 301);

        optPM.add(CreateIsCompileTimeConstantPass(true));
        optPM.add(CreateIntrinsicsOptPass());
        optPM.add(CreateInstructionSimplifyPass());
        // Currently CM does not support memset/memcpy
        // so this pass is temporary disabled for Xe.
        if (!g->target->isXeTarget()) {
            optPM.add(llvm::createMemCpyOptPass());
        }
        optPM.add(llvm::createSCCPPass());
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(CreateInstructionSimplifyPass());
        optPM.add(llvm::createJumpThreadingPass());
        optPM.add(llvm::createCorrelatedValuePropagationPass());
        optPM.add(llvm::createDeadStoreEliminationPass());
        optPM.add(llvm::createAggressiveDCEPass());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_12_0
        optPM.add(llvm::createCFGSimplificationPass(simplifyCFGopt));
#else
        optPM.add(llvm::createCFGSimplificationPass());
#endif
        optPM.add(llvm::createInstructionCombiningPass());
        optPM.add(CreateInstructionSimplifyPass());
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.add(CreateReplaceLLVMIntrinsics());
        }
#endif
        optPM.add(CreatePeepholePass());
        optPM.add(llvm::createFunctionInliningPass());
        optPM.add(llvm::createAggressiveDCEPass());
        optPM.add(llvm::createStripDeadPrototypesPass());
        optPM.add(CreateMakeInternalFuncsStaticPass());
        optPM.add(llvm::createGlobalDCEPass());
        optPM.add(llvm::createConstantMergePass());
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.add(CreateCheckUnsupportedInsts());
            optPM.add(CreateMangleOpenCLBuiltins());
            // This pass is required to prepare LLVM IR for open source SPIR-V translator
            optPM.add(
                llvm::createGenXSPIRVWriterAdaptorPass(true /*RewriteTypes*/, false /*RewriteSingleElementVectors*/));
        }
#endif
    }

    // Finish up by making sure we didn't mess anything up in the IR along
    // the way.
    optPM.add(llvm::createVerifierPass(), LAST_OPT_NUMBER);
    optPM.run(*module);

#ifndef ISPC_NO_DUMPS
    if (g->debugPrint) {
        printf("\n*****\nFINAL OUTPUT\n*****\n");
        module->dump();
    }
#endif
}

///////////////////////////////////////////////////////////////////////////
// IntrinsicsOpt

/** This is a relatively simple optimization pass that does a few small
    optimizations that LLVM's x86 optimizer doesn't currently handle.
    (Specifically, MOVMSK of a constant can be replaced with the
    corresponding constant value, BLENDVPS and AVX masked load/store with
    either an 'all on' or 'all off' masks can be replaced with simpler
    operations.

    @todo The better thing to do would be to submit a patch to LLVM to get
    these; they're presumably pretty simple patterns to match.
*/
class IntrinsicsOpt : public llvm::FunctionPass {
  public:
    IntrinsicsOpt() : FunctionPass(ID){};

    llvm::StringRef getPassName() const { return "Intrinsics Cleanup Optimization"; }

    bool runOnBasicBlock(llvm::BasicBlock &BB);

    bool runOnFunction(llvm::Function &F);

    static char ID;

  private:
    struct MaskInstruction {
        MaskInstruction(llvm::Function *f) { function = f; }
        llvm::Function *function;
    };
    std::vector<MaskInstruction> maskInstructions;

    /** Structure that records everything we need to know about a blend
        instruction for this optimization pass.
     */
    struct BlendInstruction {
        BlendInstruction(llvm::Function *f, uint64_t ao, int o0, int o1, int of)
            : function(f), allOnMask(ao), op0(o0), op1(o1), opFactor(of) {}
        /** Function pointer for the blend instruction */
        llvm::Function *function;
        /** Mask value for an "all on" mask for this instruction */
        uint64_t allOnMask;
        /** The operand number in the llvm CallInst corresponds to the
            first operand to blend with. */
        int op0;
        /** The operand number in the CallInst corresponding to the second
            operand to blend with. */
        int op1;
        /** The operand in the call inst where the blending factor is
            found. */
        int opFactor;
    };
    std::vector<BlendInstruction> blendInstructions;

    bool matchesMaskInstruction(llvm::Function *function);
    BlendInstruction *matchingBlendInstruction(llvm::Function *function);
};

char IntrinsicsOpt::ID = 0;

/** Given an llvm::Value, return true if we can determine that it's an
    undefined value.  This only makes a weak attempt at chasing this down,
    only detecting flat-out undef values, and bitcasts of undef values.

    @todo Is it worth working harder to find more of these?  It starts to
    get tricky, since having an undef operand doesn't necessarily mean that
    the result will be undefined.  (And for that matter, is there an LLVM
    call that will do this for us?)
 */
static bool lIsUndef(llvm::Value *value) {
    if (llvm::isa<llvm::UndefValue>(value))
        return true;

    llvm::BitCastInst *bci = llvm::dyn_cast<llvm::BitCastInst>(value);
    if (bci)
        return lIsUndef(bci->getOperand(0));

    return false;
}

bool IntrinsicsOpt::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("IntrinsicsOpt");

    // We can't initialize mask/blend function vector during pass initialization,
    // as they may be optimized out by the time the pass is invoked.

    // All of the mask instructions we may encounter.  Note that even if
    // compiling for AVX, we may still encounter the regular 4-wide SSE
    // MOVMSK instruction.
    if (llvm::Function *ssei8Movmsk =
            m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_sse2_pmovmskb_128))) {
        maskInstructions.push_back(ssei8Movmsk);
    }
    if (llvm::Function *sseFloatMovmsk =
            m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_sse_movmsk_ps))) {
        maskInstructions.push_back(sseFloatMovmsk);
    }
    if (llvm::Function *__movmsk = m->module->getFunction("__movmsk")) {
        maskInstructions.push_back(__movmsk);
    }
    if (llvm::Function *avxFloatMovmsk =
            m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_movmsk_ps_256))) {
        maskInstructions.push_back(avxFloatMovmsk);
    }

    // And all of the blend instructions
    blendInstructions.push_back(BlendInstruction(
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_sse41_blendvps)), 0xf, 0, 1, 2));
    blendInstructions.push_back(BlendInstruction(
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_blendv_ps_256)), 0xff, 0, 1, 2));

    llvm::Function *avxMaskedLoad32 =
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskload_ps_256));
    llvm::Function *avxMaskedLoad64 =
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskload_pd_256));
    llvm::Function *avxMaskedStore32 =
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskstore_ps_256));
    llvm::Function *avxMaskedStore64 =
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskstore_pd_256));

    bool modifiedAny = false;
restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst == NULL || callInst->getCalledFunction() == NULL)
            continue;

        BlendInstruction *blend = matchingBlendInstruction(callInst->getCalledFunction());
        if (blend != NULL) {
            llvm::Value *v[2] = {callInst->getArgOperand(blend->op0), callInst->getArgOperand(blend->op1)};
            llvm::Value *factor = callInst->getArgOperand(blend->opFactor);

            // If the values are the same, then no need to blend..
            if (v[0] == v[1]) {
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), iter, v[0]);
                modifiedAny = true;
                goto restart;
            }

            // If one of the two is undefined, we're allowed to replace
            // with the value of the other.  (In other words, the only
            // valid case is that the blend factor ends up having a value
            // that only selects from the defined one of the two operands,
            // otherwise the result is undefined and any value is fine,
            // ergo the defined one is an acceptable result.)
            if (lIsUndef(v[0])) {
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), iter, v[1]);
                modifiedAny = true;
                goto restart;
            }
            if (lIsUndef(v[1])) {
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), iter, v[0]);
                modifiedAny = true;
                goto restart;
            }

            MaskStatus maskStatus = lGetMaskStatus(factor);
            llvm::Value *value = NULL;
            if (maskStatus == MaskStatus::all_off) {
                // Mask all off -> replace with the first blend value
                value = v[0];
            } else if (maskStatus == MaskStatus::all_on) {
                // Mask all on -> replace with the second blend value
                value = v[1];
            }

            if (value != NULL) {
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), iter, value);
                modifiedAny = true;
                goto restart;
            }
        } else if (matchesMaskInstruction(callInst->getCalledFunction())) {
            llvm::Value *factor = callInst->getArgOperand(0);
            uint64_t mask;
            if (lGetMask(factor, &mask) == true) {
                // If the vector-valued mask has a known value, replace it
                // with the corresponding integer mask from its elements
                // high bits.
                llvm::Value *value = (callInst->getType() == LLVMTypes::Int32Type) ? LLVMInt32(mask) : LLVMInt64(mask);
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), iter, value);
                modifiedAny = true;
                goto restart;
            }
        } else if (callInst->getCalledFunction() == avxMaskedLoad32 ||
                   callInst->getCalledFunction() == avxMaskedLoad64) {
            llvm::Value *factor = callInst->getArgOperand(1);
            MaskStatus maskStatus = lGetMaskStatus(factor);
            if (maskStatus == MaskStatus::all_off) {
                // nothing being loaded, replace with undef value
                llvm::Type *returnType = callInst->getType();
                Assert(llvm::isa<llvm::VectorType>(returnType));
                llvm::Value *undefValue = llvm::UndefValue::get(returnType);
                llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), iter, undefValue);
                modifiedAny = true;
                goto restart;
            } else if (maskStatus == MaskStatus::all_on) {
                // all lanes active; replace with a regular load
                llvm::Type *returnType = callInst->getType();
                Assert(llvm::isa<llvm::VectorType>(returnType));
                // cast the i8 * to the appropriate type
                llvm::Value *castPtr =
                    new llvm::BitCastInst(callInst->getArgOperand(0), llvm::PointerType::get(returnType, 0),
                                          llvm::Twine(callInst->getArgOperand(0)->getName()) + "_cast", callInst);
                lCopyMetadata(castPtr, callInst);
                int align;
                if (g->opt.forceAlignedMemory)
                    align = g->target->getNativeVectorAlignment();
                else
                    align = callInst->getCalledFunction() == avxMaskedLoad32 ? 4 : 8;
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
                llvm::Instruction *loadInst =
                    new llvm::LoadInst(castPtr, llvm::Twine(callInst->getArgOperand(0)->getName()) + "_load",
                                       false /* not volatile */, llvm::MaybeAlign(align), (llvm::Instruction *)NULL);
#else
                llvm::Instruction *loadInst = new llvm::LoadInst(
                    llvm::dyn_cast<llvm::PointerType>(castPtr->getType())->getPointerElementType(), castPtr,
                    llvm::Twine(callInst->getArgOperand(0)->getName()) + "_load", false /* not volatile */,
                    llvm::MaybeAlign(align).valueOrOne(), (llvm::Instruction *)NULL);
#endif
                lCopyMetadata(loadInst, callInst);
                llvm::ReplaceInstWithInst(callInst, loadInst);
                modifiedAny = true;
                goto restart;
            }
        } else if (callInst->getCalledFunction() == avxMaskedStore32 ||
                   callInst->getCalledFunction() == avxMaskedStore64) {
            // NOTE: mask is the 2nd parameter, not the 3rd one!!
            llvm::Value *factor = callInst->getArgOperand(1);
            MaskStatus maskStatus = lGetMaskStatus(factor);
            if (maskStatus == MaskStatus::all_off) {
                // nothing actually being stored, just remove the inst
                callInst->eraseFromParent();
                modifiedAny = true;
                goto restart;
            } else if (maskStatus == MaskStatus::all_on) {
                // all lanes storing, so replace with a regular store
                llvm::Value *rvalue = callInst->getArgOperand(2);
                llvm::Type *storeType = rvalue->getType();
                llvm::Value *castPtr =
                    new llvm::BitCastInst(callInst->getArgOperand(0), llvm::PointerType::get(storeType, 0),
                                          llvm::Twine(callInst->getArgOperand(0)->getName()) + "_ptrcast", callInst);
                lCopyMetadata(castPtr, callInst);

                int align;
                if (g->opt.forceAlignedMemory)
                    align = g->target->getNativeVectorAlignment();
                else
                    align = callInst->getCalledFunction() == avxMaskedStore32 ? 4 : 8;
                llvm::StoreInst *storeInst = new llvm::StoreInst(rvalue, castPtr, (llvm::Instruction *)NULL,
                                                                 llvm::MaybeAlign(align).valueOrOne());
                lCopyMetadata(storeInst, callInst);
                llvm::ReplaceInstWithInst(callInst, storeInst);

                modifiedAny = true;
                goto restart;
            }
        }
    }

    DEBUG_END_PASS("IntrinsicsOpt");

    return modifiedAny;
}

bool IntrinsicsOpt::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("IntrinsicsOpt::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

bool IntrinsicsOpt::matchesMaskInstruction(llvm::Function *function) {
    for (unsigned int i = 0; i < maskInstructions.size(); ++i) {
        if (maskInstructions[i].function != NULL && function == maskInstructions[i].function) {
            return true;
        }
    }
    return false;
}

IntrinsicsOpt::BlendInstruction *IntrinsicsOpt::matchingBlendInstruction(llvm::Function *function) {
    for (unsigned int i = 0; i < blendInstructions.size(); ++i) {
        if (blendInstructions[i].function != NULL && function == blendInstructions[i].function) {
            return &blendInstructions[i];
        }
    }
    return NULL;
}

static llvm::Pass *CreateIntrinsicsOptPass() { return new IntrinsicsOpt; }

///////////////////////////////////////////////////////////////////////////

/** This simple optimization pass looks for a vector select instruction
    with an all-on or all-off constant mask, simplifying it to the
    appropriate operand if so.

    @todo The better thing to do would be to submit a patch to LLVM to get
    these; they're presumably pretty simple patterns to match.
*/
class InstructionSimplifyPass : public llvm::FunctionPass {
  public:
    InstructionSimplifyPass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Vector Select Optimization"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);

    static char ID;

  private:
    static bool simplifySelect(llvm::SelectInst *selectInst, llvm::BasicBlock::iterator iter);
    static llvm::Value *simplifyBoolVec(llvm::Value *value);
    static bool simplifyCall(llvm::CallInst *callInst, llvm::BasicBlock::iterator iter);
};

char InstructionSimplifyPass::ID = 0;

llvm::Value *InstructionSimplifyPass::simplifyBoolVec(llvm::Value *value) {
    llvm::TruncInst *trunc = llvm::dyn_cast<llvm::TruncInst>(value);
    if (trunc != NULL) {
        // Convert trunc({sext,zext}(i1 vector)) -> (i1 vector)
        llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(value);
        if (sext && sext->getOperand(0)->getType() == LLVMTypes::Int1VectorType)
            return sext->getOperand(0);

        llvm::ZExtInst *zext = llvm::dyn_cast<llvm::ZExtInst>(value);
        if (zext && zext->getOperand(0)->getType() == LLVMTypes::Int1VectorType)
            return zext->getOperand(0);
    }
    /*
      // This optimization has discernable benefit on the perf
      // suite on latest LLVM versions.
      // On 3.4+ (maybe even older), it can result in illegal
      // operations, so it's being disabled.
    llvm::ICmpInst *icmp = llvm::dyn_cast<llvm::ICmpInst>(value);
    if (icmp != NULL) {
        // icmp(ne, {sext,zext}(foo), zeroinitializer) -> foo
        if (icmp->getSignedPredicate() == llvm::CmpInst::ICMP_NE) {
            llvm::Value *op1 = icmp->getOperand(1);
            if (llvm::isa<llvm::ConstantAggregateZero>(op1)) {
                llvm::Value *op0 = icmp->getOperand(0);
                llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(op0);
                if (sext)
                    return sext->getOperand(0);
                llvm::ZExtInst *zext = llvm::dyn_cast<llvm::ZExtInst>(op0);
                if (zext)
                    return zext->getOperand(0);
            }
        }

    }
    */
    return NULL;
}

bool InstructionSimplifyPass::simplifySelect(llvm::SelectInst *selectInst, llvm::BasicBlock::iterator iter) {
    if (selectInst->getType()->isVectorTy() == false)
        return false;
    Assert(selectInst->getOperand(1) != NULL);
    Assert(selectInst->getOperand(2) != NULL);
    llvm::Value *factor = selectInst->getOperand(0);

    // Simplify all-on or all-off mask values
    MaskStatus maskStatus = lGetMaskStatus(factor);
    llvm::Value *value = NULL;
    if (maskStatus == MaskStatus::all_on)
        // Mask all on -> replace with the first select value
        value = selectInst->getOperand(1);
    else if (maskStatus == MaskStatus::all_off)
        // Mask all off -> replace with the second select value
        value = selectInst->getOperand(2);
    if (value != NULL) {
        llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), iter, value);
        return true;
    }

    // Sometimes earlier LLVM optimization passes generate unnecessarily
    // complex expressions for the selection vector, which in turn confuses
    // the code generators and leads to sub-optimal code (particularly for
    // 8 and 16-bit masks).  We'll try to simplify them out here so that
    // the code generator patterns match..
    if ((factor = simplifyBoolVec(factor)) != NULL) {
        llvm::Instruction *newSelect = llvm::SelectInst::Create(factor, selectInst->getOperand(1),
                                                                selectInst->getOperand(2), selectInst->getName());
        llvm::ReplaceInstWithInst(selectInst, newSelect);
        return true;
    }

    return false;
}

bool InstructionSimplifyPass::simplifyCall(llvm::CallInst *callInst, llvm::BasicBlock::iterator iter) {
    llvm::Function *calledFunc = callInst->getCalledFunction();

    // Turn a __movmsk call with a compile-time constant vector into the
    // equivalent scalar value.
    if (calledFunc == NULL || calledFunc != m->module->getFunction("__movmsk"))
        return false;

    uint64_t mask;
    if (lGetMask(callInst->getArgOperand(0), &mask) == true) {
        llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), iter, LLVMInt64(mask));
        return true;
    }
    return false;
}

bool InstructionSimplifyPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("InstructionSimplify");

    bool modifiedAny = false;

restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::SelectInst *selectInst = llvm::dyn_cast<llvm::SelectInst>(&*iter);
        if (selectInst && simplifySelect(selectInst, iter)) {
            modifiedAny = true;
            goto restart;
        }
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst && simplifyCall(callInst, iter)) {
            modifiedAny = true;
            goto restart;
        }
    }

    DEBUG_END_PASS("InstructionSimplify");

    return modifiedAny;
}

bool InstructionSimplifyPass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("InstructionSimplifyPass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateInstructionSimplifyPass() { return new InstructionSimplifyPass; }

///////////////////////////////////////////////////////////////////////////
// ImproveMemoryOpsPass

/** When the front-end emits gathers and scatters, it generates an array of
    vector-width pointers to represent the set of addresses to read from or
    write to.  This optimization detects cases when the base pointer is a
    uniform pointer or when the indexing is into an array that can be
    converted into scatters/gathers from a single base pointer and an array
    of offsets.

    See for example the comments discussing the __pseudo_gather functions
    in builtins.cpp for more information about this.
 */
class ImproveMemoryOpsPass : public llvm::FunctionPass {
  public:
    static char ID;
    ImproveMemoryOpsPass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Improve Memory Ops"; }

    bool runOnBasicBlock(llvm::BasicBlock &BB);

    bool runOnFunction(llvm::Function &F);
};

char ImproveMemoryOpsPass::ID = 0;

/** Check to make sure that this value is actually a pointer in the end.
    We need to make sure that given an expression like vec(offset) +
    ptr2int(ptr), lGetBasePointer() doesn't return vec(offset) for the base
    pointer such that we then treat ptr2int(ptr) as an offset.  This ends
    up being important so that we don't generate LLVM GEP instructions like
    "gep inttoptr 8, i64 %ptr", which in turn can lead to incorrect code
    since LLVM's pointer aliasing analysis assumes that operands after the
    first one to a GEP aren't pointers.
 */
static llvm::Value *lCheckForActualPointer(llvm::Value *v) {
    if (v == NULL) {
        return NULL;
    } else if (llvm::isa<llvm::PointerType>(v->getType())) {
        return v;
    } else if (llvm::isa<llvm::PtrToIntInst>(v)) {
        return v;
    }
    // This one is tricky, as it's heuristic tuned for LLVM 3.7+, which may
    // optimize loading double* with consequent ptr2int to straight load of i64.
    // This heuristic should be good enough to catch all the cases we should
    // detect and nothing else.
    else if (llvm::isa<llvm::LoadInst>(v)) {
        return v;
    }

    else if (llvm::CastInst *ci = llvm::dyn_cast<llvm::CastInst>(v)) {
        llvm::Value *t = lCheckForActualPointer(ci->getOperand(0));
        if (t == NULL) {
            return NULL;
        } else {
            return v;
        }
    } else {
        llvm::ConstantExpr *uce = llvm::dyn_cast<llvm::ConstantExpr>(v);
        if (uce != NULL && uce->getOpcode() == llvm::Instruction::PtrToInt)
            return v;
        return NULL;
    }
}

/** Given a llvm::Value representing a varying pointer, this function
    checks to see if all of the elements of the vector have the same value
    (i.e. there's a common base pointer). If broadcast has been already detected
    it checks that the first element of the vector is not undef. If one of the conditions
    is true, it returns the common pointer value; otherwise it returns NULL.
 */
static llvm::Value *lGetBasePointer(llvm::Value *v, llvm::Instruction *insertBefore, bool broadcastDetected) {
    if (llvm::isa<llvm::InsertElementInst>(v) || llvm::isa<llvm::ShuffleVectorInst>(v)) {
        // If we have already detected broadcast we want to look for
        // the vector with the first not-undef element
        llvm::Value *element = LLVMFlattenInsertChain(v, g->target->getVectorWidth(), true, false, broadcastDetected);
        // TODO: it's probably ok to allow undefined elements and return
        // the base pointer if all of the other elements have the same
        // value.
        if (element != NULL) {
            // all elements are the same and not NULLs
            return lCheckForActualPointer(element);
        } else {
            return NULL;
        }
    }

    // This case comes up with global/static arrays
    if (llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(v)) {
        return lCheckForActualPointer(cv->getSplatValue());
    } else if (llvm::ConstantDataVector *cdv = llvm::dyn_cast<llvm::ConstantDataVector>(v)) {
        return lCheckForActualPointer(cdv->getSplatValue());
    }
    // It is a little bit tricky to use operations with pointers, casted to int with another bit size
    // but sometimes it is useful, so we handle this case here.
    else if (llvm::CastInst *ci = llvm::dyn_cast<llvm::CastInst>(v)) {
        llvm::Value *t = lGetBasePointer(ci->getOperand(0), insertBefore, broadcastDetected);
        if (t == NULL) {
            return NULL;
        } else {
            return llvm::CastInst::Create(ci->getOpcode(), t, ci->getType()->getScalarType(),
                                          llvm::Twine(t->getName()) + "_cast", insertBefore);
        }
    }

    return NULL;
}

/** Given the two operands to a constant add expression, see if we have the
    form "base pointer + offset", whee op0 is the base pointer and op1 is
    the offset; if so return the base and the offset. */
static llvm::Constant *lGetConstantAddExprBaseOffset(llvm::Constant *op0, llvm::Constant *op1, llvm::Constant **delta) {
    llvm::ConstantExpr *op = llvm::dyn_cast<llvm::ConstantExpr>(op0);
    if (op == NULL || op->getOpcode() != llvm::Instruction::PtrToInt)
        // the first operand isn't a pointer
        return NULL;

    llvm::ConstantInt *opDelta = llvm::dyn_cast<llvm::ConstantInt>(op1);
    if (opDelta == NULL)
        // the second operand isn't an integer operand
        return NULL;

    *delta = opDelta;
    return op0;
}

static llvm::Value *lExtractFromInserts(llvm::Value *v, unsigned int index) {
    llvm::InsertValueInst *iv = llvm::dyn_cast<llvm::InsertValueInst>(v);
    if (iv == NULL)
        return NULL;

    Assert(iv->hasIndices() && iv->getNumIndices() == 1);
    if (iv->getIndices()[0] == index)
        return iv->getInsertedValueOperand();
    else
        return lExtractFromInserts(iv->getAggregateOperand(), index);
}

/** Given a varying pointer in ptrs, this function checks to see if it can
    be determined to be indexing from a common uniform base pointer.  If
    so, the function returns the base pointer llvm::Value and initializes
    *offsets with an int vector of the per-lane offsets
 */
static llvm::Value *lGetBasePtrAndOffsets(llvm::Value *ptrs, llvm::Value **offsets, llvm::Instruction *insertBefore) {
#ifndef ISPC_NO_DUMPS
    if (g->debugPrint) {
        fprintf(stderr, "lGetBasePtrAndOffsets\n");
        LLVMDumpValue(ptrs);
    }
#endif

    bool broadcastDetected = false;
    // Looking for %gep_offset = shufflevector <8 x i64> %0, <8 x i64> undef, <8 x i32> zeroinitializer
    llvm::ShuffleVectorInst *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(ptrs);
    if (shuffle != NULL) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
        llvm::Value *indices = shuffle->getShuffleMaskForBitcode();
#else
        llvm::Value *indices = shuffle->getOperand(2);
#endif
        llvm::Value *vec = shuffle->getOperand(1);

        if (lIsUndef(vec) && llvm::isa<llvm::ConstantAggregateZero>(indices)) {
            broadcastDetected = true;
        }
    }
    llvm::Value *base = lGetBasePointer(ptrs, insertBefore, broadcastDetected);
    if (base != NULL) {
        // We have a straight up varying pointer with no indexing that's
        // actually all the same value.
        if (g->target->is32Bit())
            *offsets = LLVMInt32Vector(0);
        else
            *offsets = LLVMInt64Vector((int64_t)0);

        if (broadcastDetected) {
            llvm::Value *op = shuffle->getOperand(0);
            llvm::BinaryOperator *bop_var = llvm::dyn_cast<llvm::BinaryOperator>(op);
            if (bop_var != NULL && ((bop_var->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop_var))) {
                // We expect here ConstantVector as
                // <i64 4, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>
                llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(bop_var->getOperand(1));
                llvm::Instruction *shuffle_offset = NULL;
                if (cv != NULL) {
                    llvm::Value *zeroMask =
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
                        llvm::ConstantVector::getSplat(cv->getType()->getVectorNumElements(),
#elif ISPC_LLVM_VERSION < ISPC_LLVM_12_0
                        llvm::ConstantVector::getSplat(
                            {llvm::dyn_cast<llvm::VectorType>(cv->getType())->getNumElements(), false},
#else // LLVM 12.0+
                        llvm::ConstantVector::getSplat(
                            llvm::ElementCount::get(
                                llvm::dyn_cast<llvm::FixedVectorType>(cv->getType())->getNumElements(), false),
#endif
                                                       llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*g->ctx)));
                    // Create offset
                    shuffle_offset = new llvm::ShuffleVectorInst(cv, llvm::UndefValue::get(cv->getType()), zeroMask,
                                                                 "shuffle", bop_var);
                } else {
                    // or it binaryoperator can accept another binary operator
                    // that is a result of counting another part of offset:
                    // %another_bop = bop <16 x i32> %vec, <i32 7, i32 undef, i32 undef, ...>
                    // %offsets = add <16 x i32> %another_bop, %base
                    bop_var = llvm::dyn_cast<llvm::BinaryOperator>(bop_var->getOperand(0));
                    if (bop_var != NULL) {
                        llvm::Type *bop_var_type = bop_var->getType();
                        llvm::Value *zeroMask = llvm::ConstantVector::getSplat(
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
                            bop_var_type->getVectorNumElements(),
#elif ISPC_LLVM_VERSION < ISPC_LLVM_12_0
                            {llvm::dyn_cast<llvm::VectorType>(bop_var_type)->getNumElements(), false},
#else // LLVM 12.0+
                            llvm::ElementCount::get(
                                llvm::dyn_cast<llvm::FixedVectorType>(bop_var_type)->getNumElements(), false),
#endif
                            llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*g->ctx)));
                        shuffle_offset = new llvm::ShuffleVectorInst(bop_var, llvm::UndefValue::get(bop_var_type),
                                                                     zeroMask, "shuffle");
                        shuffle_offset->insertAfter(bop_var);
                    }
                }
                if (shuffle_offset != NULL) {
                    *offsets = llvm::BinaryOperator::Create(llvm::Instruction::Add, *offsets, shuffle_offset,
                                                            "new_offsets", insertBefore);
                    return base;
                } else {
                    // Base + offset pattern was not recognized
                    return NULL;
                }
            }
        }
        return base;
    }

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(ptrs);
    if (bop != NULL && ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop))) {
        // If we have a common pointer plus something, then we're also
        // good.
        if ((base = lGetBasePtrAndOffsets(bop->getOperand(0), offsets, insertBefore)) != NULL) {
            *offsets = llvm::BinaryOperator::Create(llvm::Instruction::Add, *offsets, bop->getOperand(1), "new_offsets",
                                                    insertBefore);
            return base;
        } else if ((base = lGetBasePtrAndOffsets(bop->getOperand(1), offsets, insertBefore)) != NULL) {
            *offsets = llvm::BinaryOperator::Create(llvm::Instruction::Add, *offsets, bop->getOperand(0), "new_offsets",
                                                    insertBefore);
            return base;
        }
    }
    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(ptrs);
    if (cv != NULL) {
        // Indexing into global arrays can lead to this form, with
        // ConstantVectors..
        llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> elements;
        for (int i = 0; i < (int)cv->getNumOperands(); ++i) {
            llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(cv->getOperand(i));
            if (c == NULL)
                return NULL;
            elements.push_back(c);
        }

        llvm::Constant *delta[ISPC_MAX_NVEC];
        for (unsigned int i = 0; i < elements.size(); ++i) {
            // For each element, try to decompose it into either a straight
            // up base pointer, or a base pointer plus an integer value.
            llvm::ConstantExpr *ce = llvm::dyn_cast<llvm::ConstantExpr>(elements[i]);
            if (ce == NULL)
                return NULL;

            delta[i] = NULL;
            llvm::Value *elementBase = NULL; // base pointer for this element
            if (ce->getOpcode() == llvm::Instruction::PtrToInt) {
                // If the element is just a ptr to int instruction, treat
                // it as having an offset of zero
                elementBase = ce;
                delta[i] = g->target->is32Bit() ? LLVMInt32(0) : LLVMInt64(0);
            } else if ((ce->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(ce)) {
                // Try both orderings of the operands to see if we can get
                // a pointer+offset out of them.
                elementBase = lGetConstantAddExprBaseOffset(ce->getOperand(0), ce->getOperand(1), &delta[i]);
                if (elementBase == NULL)
                    elementBase = lGetConstantAddExprBaseOffset(ce->getOperand(1), ce->getOperand(0), &delta[i]);
            }

            // We weren't able to find a base pointer in the above.  (We
            // don't expect this to happen; if it does, it may be necessary
            // to handle more cases in the decomposition above.)
            if (elementBase == NULL)
                return NULL;

            Assert(delta[i] != NULL);
            if (base == NULL)
                // The first time we've found a base pointer
                base = elementBase;
            else if (base != elementBase)
                // Different program instances have different base
                // pointers, so no luck.
                return NULL;
        }

        Assert(base != NULL);
        llvm::ArrayRef<llvm::Constant *> deltas(&delta[0], &delta[elements.size()]);
        *offsets = llvm::ConstantVector::get(deltas);
        return base;
    }

    llvm::ExtractValueInst *ev = llvm::dyn_cast<llvm::ExtractValueInst>(ptrs);
    if (ev != NULL) {
        Assert(ev->getNumIndices() == 1);
        int index = ev->getIndices()[0];
        ptrs = lExtractFromInserts(ev->getAggregateOperand(), index);
        if (ptrs != NULL)
            return lGetBasePtrAndOffsets(ptrs, offsets, insertBefore);
    }

    return NULL;
}

/** Given a vector expression in vec, separate it into a compile-time
    constant component and a variable component, returning the two parts in
    *constOffset and *variableOffset.  (It should be the case that the sum
    of these two is exactly equal to the original vector.)

    This routine only handles some (important) patterns; in some cases it
    will fail and return components that are actually compile-time
    constants in *variableOffset.

    Finally, if there aren't any constant (or, respectivaly, variable)
    components, the corresponding return value may be set to NULL.
 */
static void lExtractConstantOffset(llvm::Value *vec, llvm::Value **constOffset, llvm::Value **variableOffset,
                                   llvm::Instruction *insertBefore) {
    if (llvm::isa<llvm::ConstantVector>(vec) || llvm::isa<llvm::ConstantDataVector>(vec) ||
        llvm::isa<llvm::ConstantAggregateZero>(vec)) {
        *constOffset = vec;
        *variableOffset = NULL;
        return;
    }

    llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(vec);
    if (cast != NULL) {
        // Check the cast target.
        llvm::Value *co, *vo;
        lExtractConstantOffset(cast->getOperand(0), &co, &vo, insertBefore);

        // make new cast instructions for the two parts
        if (co == NULL)
            *constOffset = NULL;
        else
            *constOffset = llvm::CastInst::Create(cast->getOpcode(), co, cast->getType(),
                                                  llvm::Twine(co->getName()) + "_cast", insertBefore);
        if (vo == NULL)
            *variableOffset = NULL;
        else
            *variableOffset = llvm::CastInst::Create(cast->getOpcode(), vo, cast->getType(),
                                                     llvm::Twine(vo->getName()) + "_cast", insertBefore);
        return;
    }

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(vec);
    if (bop != NULL) {
        llvm::Value *op0 = bop->getOperand(0);
        llvm::Value *op1 = bop->getOperand(1);
        llvm::Value *c0, *v0, *c1, *v1;

        if ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop)) {
            lExtractConstantOffset(op0, &c0, &v0, insertBefore);
            lExtractConstantOffset(op1, &c1, &v1, insertBefore);

            if (c0 == NULL || llvm::isa<llvm::ConstantAggregateZero>(c0))
                *constOffset = c1;
            else if (c1 == NULL || llvm::isa<llvm::ConstantAggregateZero>(c1))
                *constOffset = c0;
            else
                *constOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Add, c0, c1, ((llvm::Twine("add_") + c0->getName()) + "_") + c1->getName(),
                    insertBefore);

            if (v0 == NULL || llvm::isa<llvm::ConstantAggregateZero>(v0))
                *variableOffset = v1;
            else if (v1 == NULL || llvm::isa<llvm::ConstantAggregateZero>(v1))
                *variableOffset = v0;
            else
                *variableOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Add, v0, v1, ((llvm::Twine("add_") + v0->getName()) + "_") + v1->getName(),
                    insertBefore);
            return;
        } else if (bop->getOpcode() == llvm::Instruction::Shl) {
            lExtractConstantOffset(op0, &c0, &v0, insertBefore);
            lExtractConstantOffset(op1, &c1, &v1, insertBefore);

            // Given the product of constant and variable terms, we have:
            // (c0 + v0) * (2^(c1 + v1))  = c0 * 2^c1 * 2^v1 + v0 * 2^c1 * 2^v1
            // We can optimize only if v1 == NULL.
            if ((v1 != NULL) || (c0 == NULL) || (c1 == NULL)) {
                *constOffset = NULL;
                *variableOffset = vec;
            } else if (v0 == NULL) {
                *constOffset = vec;
                *variableOffset = NULL;
            } else {
                *constOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Shl, c0, c1, ((llvm::Twine("shl_") + c0->getName()) + "_") + c1->getName(),
                    insertBefore);
                *variableOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Shl, v0, c1, ((llvm::Twine("shl_") + v0->getName()) + "_") + c1->getName(),
                    insertBefore);
            }
            return;
        } else if (bop->getOpcode() == llvm::Instruction::Mul) {
            lExtractConstantOffset(op0, &c0, &v0, insertBefore);
            lExtractConstantOffset(op1, &c1, &v1, insertBefore);

            // Given the product of constant and variable terms, we have:
            // (c0 + v0) * (c1 + v1) == (c0 c1) + (v0 c1 + c0 v1 + v0 v1)
            // Note that the first term is a constant and the last three are
            // variable.
            if (c0 != NULL && c1 != NULL)
                *constOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Mul, c0, c1, ((llvm::Twine("mul_") + c0->getName()) + "_") + c1->getName(),
                    insertBefore);
            else
                *constOffset = NULL;

            llvm::Value *va = NULL, *vb = NULL, *vc = NULL;
            if (v0 != NULL && c1 != NULL)
                va = llvm::BinaryOperator::Create(llvm::Instruction::Mul, v0, c1,
                                                  ((llvm::Twine("mul_") + v0->getName()) + "_") + c1->getName(),
                                                  insertBefore);
            if (c0 != NULL && v1 != NULL)
                vb = llvm::BinaryOperator::Create(llvm::Instruction::Mul, c0, v1,
                                                  ((llvm::Twine("mul_") + c0->getName()) + "_") + v1->getName(),
                                                  insertBefore);
            if (v0 != NULL && v1 != NULL)
                vc = llvm::BinaryOperator::Create(llvm::Instruction::Mul, v0, v1,
                                                  ((llvm::Twine("mul_") + v0->getName()) + "_") + v1->getName(),
                                                  insertBefore);

            llvm::Value *vab = NULL;
            if (va != NULL && vb != NULL)
                vab = llvm::BinaryOperator::Create(llvm::Instruction::Add, va, vb,
                                                   ((llvm::Twine("add_") + va->getName()) + "_") + vb->getName(),
                                                   insertBefore);
            else if (va != NULL)
                vab = va;
            else
                vab = vb;

            if (vab != NULL && vc != NULL)
                *variableOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Add, vab, vc, ((llvm::Twine("add_") + vab->getName()) + "_") + vc->getName(),
                    insertBefore);
            else if (vab != NULL)
                *variableOffset = vab;
            else
                *variableOffset = vc;

            return;
        }
    }

    // Nothing matched, just return what we have as a variable component
    *constOffset = NULL;
    *variableOffset = vec;
}

/* Returns true if the given value is a constant vector of integers with
   the same value in all of the elements.  (Returns the splatted value in
   *splat, if so). */
static bool lIsIntegerSplat(llvm::Value *v, int *splat) {
    llvm::ConstantDataVector *cvec = llvm::dyn_cast<llvm::ConstantDataVector>(v);
    if (cvec == NULL)
        return false;

    llvm::Constant *splatConst = cvec->getSplatValue();
    if (splatConst == NULL)
        return false;

    llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(splatConst);
    if (ci == NULL)
        return false;

    int64_t splatVal = ci->getSExtValue();
    *splat = (int)splatVal;
    return true;
}

static llvm::Value *lExtract248Scale(llvm::Value *splatOperand, int splatValue, llvm::Value *otherOperand,
                                     llvm::Value **result) {
    if (splatValue == 2 || splatValue == 4 || splatValue == 8) {
        *result = otherOperand;
        return LLVMInt32(splatValue);
    }
    // Even if we don't have a common scale by exactly 2, 4, or 8, we'll
    // see if we can pull out that much of the scale anyway; this may in
    // turn allow other optimizations later.
    for (int scale = 8; scale >= 2; scale /= 2) {
        llvm::Instruction *insertBefore = llvm::dyn_cast<llvm::Instruction>(*result);
        Assert(insertBefore != NULL);

        if ((splatValue % scale) == 0) {
            // *result = otherOperand * splatOperand / scale;
            llvm::Value *splatScaleVec = (splatOperand->getType() == LLVMTypes::Int32VectorType)
                                             ? LLVMInt32Vector(scale)
                                             : LLVMInt64Vector(scale);
            llvm::Value *splatDiv =
                llvm::BinaryOperator::Create(llvm::Instruction::SDiv, splatOperand, splatScaleVec, "div", insertBefore);
            *result = llvm::BinaryOperator::Create(llvm::Instruction::Mul, splatDiv, otherOperand, "mul", insertBefore);
            return LLVMInt32(scale);
        }
    }
    return LLVMInt32(1);
}

/** Given a vector of integer offsets to a base pointer being used for a
    gather or a scatter, see if its root operation is a multiply by a
    vector of some value by all 2s/4s/8s.  If not, return NULL.

    If it is return an i32 value of 2, 4, 8 from the function and modify
    *vec so that it points to the operand that is being multiplied by
    2/4/8.

    We go through all this trouble so that we can pass the i32 scale factor
    to the {gather,scatter}_base_offsets function as a separate scale
    factor for the offsets.  This in turn is used in a way so that the LLVM
    x86 code generator matches it to apply x86's free scale by 2x, 4x, or
    8x to one of two registers being added together for an addressing
    calculation.
 */
static llvm::Value *lExtractOffsetVector248Scale(llvm::Value **vec) {
    llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(*vec);
    if (cast != NULL) {
        llvm::Value *castOp = cast->getOperand(0);
        // Check the cast target.
        llvm::Value *scale = lExtractOffsetVector248Scale(&castOp);
        if (scale == NULL)
            return NULL;

        // make a new cast instruction so that we end up with the right
        // type
        *vec = llvm::CastInst::Create(cast->getOpcode(), castOp, cast->getType(), "offset_cast", cast);
        return scale;
    }

    // If we don't have a binary operator, then just give up
    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(*vec);
    if (bop == NULL)
        return LLVMInt32(1);

    llvm::Value *op0 = bop->getOperand(0), *op1 = bop->getOperand(1);
    if ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop)) {
        if (llvm::isa<llvm::ConstantAggregateZero>(op0)) {
            *vec = op1;
            return lExtractOffsetVector248Scale(vec);
        } else if (llvm::isa<llvm::ConstantAggregateZero>(op1)) {
            *vec = op0;
            return lExtractOffsetVector248Scale(vec);
        } else {
            llvm::Value *s0 = lExtractOffsetVector248Scale(&op0);
            llvm::Value *s1 = lExtractOffsetVector248Scale(&op1);
            if (s0 == s1) {
                *vec = llvm::BinaryOperator::Create(llvm::Instruction::Add, op0, op1, "new_add", bop);
                return s0;
            } else
                return LLVMInt32(1);
        }
    } else if (bop->getOpcode() == llvm::Instruction::Mul) {
        // Check each operand for being one of the scale factors we care about.
        int splat;
        if (lIsIntegerSplat(op0, &splat))
            return lExtract248Scale(op0, splat, op1, vec);
        else if (lIsIntegerSplat(op1, &splat))
            return lExtract248Scale(op1, splat, op0, vec);
        else
            return LLVMInt32(1);
    } else
        return LLVMInt32(1);
}

#if 0
static llvm::Value *
lExtractUniforms(llvm::Value **vec, llvm::Instruction *insertBefore) {
    fprintf(stderr, " lextract: ");
    (*vec)->dump();
    fprintf(stderr, "\n");

    if (llvm::isa<llvm::ConstantVector>(*vec) ||
        llvm::isa<llvm::ConstantDataVector>(*vec) ||
        llvm::isa<llvm::ConstantAggregateZero>(*vec))
        return NULL;

    llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(*vec);
    if (sext != NULL) {
        llvm::Value *sextOp = sext->getOperand(0);
        // Check the sext target.
        llvm::Value *unif = lExtractUniforms(&sextOp, insertBefore);
        if (unif == NULL)
            return NULL;

        // make a new sext instruction so that we end up with the right
        // type
        *vec = new llvm::SExtInst(sextOp, sext->getType(), "offset_sext", sext);
        return unif;
    }

    if (LLVMVectorValuesAllEqual(*vec)) {
        // FIXME: we may want to redo all of the expression here, in scalar
        // form (if at all possible), for code quality...
        llvm::Value *unif =
            llvm::ExtractElementInst::Create(*vec, LLVMInt32(0),
                                             "first_uniform", insertBefore);
        *vec = NULL;
        return unif;
    }

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(*vec);
    if (bop == NULL)
        return NULL;

    llvm::Value *op0 = bop->getOperand(0), *op1 = bop->getOperand(1);
    if (bop->getOpcode() == llvm::Instruction::Add) {
        llvm::Value *s0 = lExtractUniforms(&op0, insertBefore);
        llvm::Value *s1 = lExtractUniforms(&op1, insertBefore);
        if (s0 == NULL && s1 == NULL)
            return NULL;

        if (op0 == NULL)
            *vec = op1;
        else if (op1 == NULL)
            *vec = op0;
        else
            *vec = llvm::BinaryOperator::Create(llvm::Instruction::Add,
                                                op0, op1, "new_add", insertBefore);

        if (s0 == NULL)
            return s1;
        else if (s1 == NULL)
            return s0;
        else
            return llvm::BinaryOperator::Create(llvm::Instruction::Add, s0, s1,
                                                "add_unif", insertBefore);
    }
#if 0
    else if (bop->getOpcode() == llvm::Instruction::Mul) {
        // Check each operand for being one of the scale factors we care about.
        int splat;
        if (lIs248Splat(op0, &splat)) {
            *vec = op1;
            return LLVMInt32(splat);
        }
        else if (lIs248Splat(op1, &splat)) {
            *vec = op0;
            return LLVMInt32(splat);
        }
        else
            return LLVMInt32(1);
    }
#endif
    else
        return NULL;
}


static void
lExtractUniformsFromOffset(llvm::Value **basePtr, llvm::Value **offsetVector,
                           llvm::Value *offsetScale,
                           llvm::Instruction *insertBefore) {
#if 1
    (*basePtr)->dump();
    printf("\n");
    (*offsetVector)->dump();
    printf("\n");
    offsetScale->dump();
    printf("-----\n");
#endif

    llvm::Value *uniformDelta = lExtractUniforms(offsetVector, insertBefore);
    if (uniformDelta == NULL)
        return;

    *basePtr = lGEPInst(*basePtr, arrayRef, "new_base", insertBefore);

    // this should only happen if we have only uniforms, but that in turn
    // shouldn't be a gather/scatter!
    Assert(*offsetVector != NULL);
}
#endif

static bool lVectorIs32BitInts(llvm::Value *v) {
    int nElts;
    int64_t elts[ISPC_MAX_NVEC];
    if (!LLVMExtractVectorInts(v, elts, &nElts))
        return false;

    for (int i = 0; i < nElts; ++i)
        if ((int32_t)elts[i] != elts[i])
            return false;

    return true;
}

/** Check to see if the two offset vectors can safely be represented with
    32-bit values.  If so, return true and update the pointed-to
    llvm::Value *s to be the 32-bit equivalents. */
static bool lOffsets32BitSafe(llvm::Value **variableOffsetPtr, llvm::Value **constOffsetPtr,
                              llvm::Instruction *insertBefore) {
    llvm::Value *variableOffset = *variableOffsetPtr;
    llvm::Value *constOffset = *constOffsetPtr;

    if (variableOffset->getType() != LLVMTypes::Int32VectorType) {
        llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(variableOffset);
        if (sext != NULL && sext->getOperand(0)->getType() == LLVMTypes::Int32VectorType)
            // sext of a 32-bit vector -> the 32-bit vector is good
            variableOffset = sext->getOperand(0);
        else if (lVectorIs32BitInts(variableOffset))
            // The only constant vector we should have here is a vector of
            // all zeros (i.e. a ConstantAggregateZero, but just in case,
            // do the more general check with lVectorIs32BitInts().
            variableOffset = new llvm::TruncInst(variableOffset, LLVMTypes::Int32VectorType,
                                                 llvm::Twine(variableOffset->getName()) + "_trunc", insertBefore);
        else
            return false;
    }

    if (constOffset->getType() != LLVMTypes::Int32VectorType) {
        if (lVectorIs32BitInts(constOffset)) {
            // Truncate them so we have a 32-bit vector type for them.
            constOffset = new llvm::TruncInst(constOffset, LLVMTypes::Int32VectorType,
                                              llvm::Twine(constOffset->getName()) + "_trunc", insertBefore);
        } else {
            // FIXME: otherwise we just assume that all constant offsets
            // can actually always fit into 32-bits...  (This could be
            // wrong, but it should be only in pretty esoteric cases).  We
            // make this assumption for now since we sometimes generate
            // constants that need constant folding before we really have a
            // constant vector out of them, and
            // llvm::ConstantFoldInstruction() doesn't seem to be doing
            // enough for us in some cases if we call it from here.
            constOffset = new llvm::TruncInst(constOffset, LLVMTypes::Int32VectorType,
                                              llvm::Twine(constOffset->getName()) + "_trunc", insertBefore);
        }
    }

    *variableOffsetPtr = variableOffset;
    *constOffsetPtr = constOffset;
    return true;
}

/** Check to see if the offset value is composed of a string of Adds,
    SExts, and Constant Vectors that are 32-bit safe.  Recursively
    explores the operands of Add instructions (as they might themselves
    be adds that eventually terminate in constant vectors or a SExt.)
 */

static bool lIs32BitSafeHelper(llvm::Value *v) {
    // handle Adds, SExts, Constant Vectors
    if (llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v)) {
        if ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop)) {
            return lIs32BitSafeHelper(bop->getOperand(0)) && lIs32BitSafeHelper(bop->getOperand(1));
        }
        return false;
    } else if (llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(v)) {
        return sext->getOperand(0)->getType() == LLVMTypes::Int32VectorType;
    } else
        return lVectorIs32BitInts(v);
}

/** Check to see if the single offset vector can safely be represented with
    32-bit values.  If so, return true and update the pointed-to
    llvm::Value * to be the 32-bit equivalent. */
static bool lOffsets32BitSafe(llvm::Value **offsetPtr, llvm::Instruction *insertBefore) {
    llvm::Value *offset = *offsetPtr;

    if (offset->getType() == LLVMTypes::Int32VectorType)
        return true;

    llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(offset);
    if (sext != NULL && sext->getOperand(0)->getType() == LLVMTypes::Int32VectorType) {
        // sext of a 32-bit vector -> the 32-bit vector is good
        *offsetPtr = sext->getOperand(0);
        return true;
    } else if (lIs32BitSafeHelper(offset)) {
        // The only constant vector we should have here is a vector of
        // all zeros (i.e. a ConstantAggregateZero, but just in case,
        // do the more general check with lVectorIs32BitInts().

        // Alternatively, offset could be a sequence of adds terminating
        // in safe constant vectors or a SExt.
        *offsetPtr = new llvm::TruncInst(offset, LLVMTypes::Int32VectorType, llvm::Twine(offset->getName()) + "_trunc",
                                         insertBefore);
        return true;
    } else
        return false;
}

static bool lGSToGSBaseOffsets(llvm::CallInst *callInst) {
    struct GSInfo {
        GSInfo(const char *pgFuncName, const char *pgboFuncName, const char *pgbo32FuncName, bool ig, bool ip)
            : isGather(ig), isPrefetch(ip) {
            func = m->module->getFunction(pgFuncName);
            baseOffsetsFunc = m->module->getFunction(pgboFuncName);
            baseOffsets32Func = m->module->getFunction(pgbo32FuncName);
        }
        llvm::Function *func;
        llvm::Function *baseOffsetsFunc, *baseOffsets32Func;
        const bool isGather;
        const bool isPrefetch;
    };

    GSInfo gsFuncs[] = {
        GSInfo(
            "__pseudo_gather32_i8",
            g->target->hasGather() ? "__pseudo_gather_base_offsets32_i8" : "__pseudo_gather_factored_base_offsets32_i8",
            g->target->hasGather() ? "__pseudo_gather_base_offsets32_i8" : "__pseudo_gather_factored_base_offsets32_i8",
            true, false),
        GSInfo("__pseudo_gather32_i16",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_i16"
                                      : "__pseudo_gather_factored_base_offsets32_i16",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_i16"
                                      : "__pseudo_gather_factored_base_offsets32_i16",
               true, false),
        GSInfo("__pseudo_gather32_half",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_half"
                                      : "__pseudo_gather_factored_base_offsets32_half",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_half"
                                      : "__pseudo_gather_factored_base_offsets32_half",
               true, false),
        GSInfo("__pseudo_gather32_i32",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_i32"
                                      : "__pseudo_gather_factored_base_offsets32_i32",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_i32"
                                      : "__pseudo_gather_factored_base_offsets32_i32",
               true, false),
        GSInfo("__pseudo_gather32_float",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_float"
                                      : "__pseudo_gather_factored_base_offsets32_float",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_float"
                                      : "__pseudo_gather_factored_base_offsets32_float",
               true, false),
        GSInfo("__pseudo_gather32_i64",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_i64"
                                      : "__pseudo_gather_factored_base_offsets32_i64",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_i64"
                                      : "__pseudo_gather_factored_base_offsets32_i64",
               true, false),
        GSInfo("__pseudo_gather32_double",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_double"
                                      : "__pseudo_gather_factored_base_offsets32_double",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_double"
                                      : "__pseudo_gather_factored_base_offsets32_double",
               true, false),

        GSInfo("__pseudo_scatter32_i8",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i8"
                                       : "__pseudo_scatter_factored_base_offsets32_i8",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i8"
                                       : "__pseudo_scatter_factored_base_offsets32_i8",
               false, false),
        GSInfo("__pseudo_scatter32_i16",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i16"
                                       : "__pseudo_scatter_factored_base_offsets32_i16",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i16"
                                       : "__pseudo_scatter_factored_base_offsets32_i16",
               false, false),
        GSInfo("__pseudo_scatter32_half",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_half"
                                       : "__pseudo_scatter_factored_base_offsets32_half",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_half"
                                       : "__pseudo_scatter_factored_base_offsets32_half",
               false, false),
        GSInfo("__pseudo_scatter32_i32",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i32"
                                       : "__pseudo_scatter_factored_base_offsets32_i32",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i32"
                                       : "__pseudo_scatter_factored_base_offsets32_i32",
               false, false),
        GSInfo("__pseudo_scatter32_float",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_float"
                                       : "__pseudo_scatter_factored_base_offsets32_float",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_float"
                                       : "__pseudo_scatter_factored_base_offsets32_float",
               false, false),
        GSInfo("__pseudo_scatter32_i64",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i64"
                                       : "__pseudo_scatter_factored_base_offsets32_i64",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i64"
                                       : "__pseudo_scatter_factored_base_offsets32_i64",
               false, false),
        GSInfo("__pseudo_scatter32_double",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_double"
                                       : "__pseudo_scatter_factored_base_offsets32_double",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_double"
                                       : "__pseudo_scatter_factored_base_offsets32_double",
               false, false),

        GSInfo(
            "__pseudo_gather64_i8",
            g->target->hasGather() ? "__pseudo_gather_base_offsets64_i8" : "__pseudo_gather_factored_base_offsets64_i8",
            g->target->hasGather() ? "__pseudo_gather_base_offsets32_i8" : "__pseudo_gather_factored_base_offsets32_i8",
            true, false),
        GSInfo("__pseudo_gather64_i16",
               g->target->hasGather() ? "__pseudo_gather_base_offsets64_i16"
                                      : "__pseudo_gather_factored_base_offsets64_i16",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_i16"
                                      : "__pseudo_gather_factored_base_offsets32_i16",
               true, false),
        GSInfo("__pseudo_gather64_half",
               g->target->hasGather() ? "__pseudo_gather_base_offsets64_half"
                                      : "__pseudo_gather_factored_base_offsets64_half",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_half"
                                      : "__pseudo_gather_factored_base_offsets32_half",
               true, false),
        GSInfo("__pseudo_gather64_i32",
               g->target->hasGather() ? "__pseudo_gather_base_offsets64_i32"
                                      : "__pseudo_gather_factored_base_offsets64_i32",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_i32"
                                      : "__pseudo_gather_factored_base_offsets32_i32",
               true, false),
        GSInfo("__pseudo_gather64_float",
               g->target->hasGather() ? "__pseudo_gather_base_offsets64_float"
                                      : "__pseudo_gather_factored_base_offsets64_float",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_float"
                                      : "__pseudo_gather_factored_base_offsets32_float",
               true, false),
        GSInfo("__pseudo_gather64_i64",
               g->target->hasGather() ? "__pseudo_gather_base_offsets64_i64"
                                      : "__pseudo_gather_factored_base_offsets64_i64",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_i64"
                                      : "__pseudo_gather_factored_base_offsets32_i64",
               true, false),
        GSInfo("__pseudo_gather64_double",
               g->target->hasGather() ? "__pseudo_gather_base_offsets64_double"
                                      : "__pseudo_gather_factored_base_offsets64_double",
               g->target->hasGather() ? "__pseudo_gather_base_offsets32_double"
                                      : "__pseudo_gather_factored_base_offsets32_double",
               true, false),

        GSInfo("__pseudo_scatter64_i8",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_i8"
                                       : "__pseudo_scatter_factored_base_offsets64_i8",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i8"
                                       : "__pseudo_scatter_factored_base_offsets32_i8",
               false, false),
        GSInfo("__pseudo_scatter64_i16",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_i16"
                                       : "__pseudo_scatter_factored_base_offsets64_i16",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i16"
                                       : "__pseudo_scatter_factored_base_offsets32_i16",
               false, false),
        GSInfo("__pseudo_scatter64_half",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_half"
                                       : "__pseudo_scatter_factored_base_offsets64_half",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_half"
                                       : "__pseudo_scatter_factored_base_offsets32_half",
               false, false),
        GSInfo("__pseudo_scatter64_i32",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_i32"
                                       : "__pseudo_scatter_factored_base_offsets64_i32",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i32"
                                       : "__pseudo_scatter_factored_base_offsets32_i32",
               false, false),
        GSInfo("__pseudo_scatter64_float",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_float"
                                       : "__pseudo_scatter_factored_base_offsets64_float",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_float"
                                       : "__pseudo_scatter_factored_base_offsets32_float",
               false, false),
        GSInfo("__pseudo_scatter64_i64",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_i64"
                                       : "__pseudo_scatter_factored_base_offsets64_i64",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i64"
                                       : "__pseudo_scatter_factored_base_offsets32_i64",
               false, false),
        GSInfo("__pseudo_scatter64_double",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_double"
                                       : "__pseudo_scatter_factored_base_offsets64_double",
               g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_double"
                                       : "__pseudo_scatter_factored_base_offsets32_double",
               false, false),
        GSInfo("__pseudo_prefetch_read_varying_1",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_1_native" : "__prefetch_read_varying_1",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_1_native" : "__prefetch_read_varying_1",
               false, true),

        GSInfo("__pseudo_prefetch_read_varying_2",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_2_native" : "__prefetch_read_varying_2",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_2_native" : "__prefetch_read_varying_2",
               false, true),

        GSInfo("__pseudo_prefetch_read_varying_3",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_3_native" : "__prefetch_read_varying_3",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_3_native" : "__prefetch_read_varying_3",
               false, true),

        GSInfo("__pseudo_prefetch_read_varying_nt",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_nt_native" : "__prefetch_read_varying_nt",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_nt_native" : "__prefetch_read_varying_nt",
               false, true),

        GSInfo("__pseudo_prefetch_write_varying_1",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_1_native" : "__prefetch_write_varying_1",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_1_native" : "__prefetch_write_varying_1",
               false, true),
        GSInfo("__pseudo_prefetch_write_varying_2",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_2_native" : "__prefetch_write_varying_2",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_2_native" : "__prefetch_write_varying_2",
               false, true),

        GSInfo("__pseudo_prefetch_write_varying_3",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_3_native" : "__prefetch_write_varying_3",
               g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_3_native" : "__prefetch_write_varying_3",
               false, true),
    };

    int numGSFuncs = sizeof(gsFuncs) / sizeof(gsFuncs[0]);
    for (int i = 0; i < numGSFuncs; ++i)
        Assert(gsFuncs[i].func != NULL && gsFuncs[i].baseOffsetsFunc != NULL && gsFuncs[i].baseOffsets32Func != NULL);

    GSInfo *info = NULL;
    for (int i = 0; i < numGSFuncs; ++i)
        if (gsFuncs[i].func != NULL && callInst->getCalledFunction() == gsFuncs[i].func) {
            info = &gsFuncs[i];
            break;
        }
    if (info == NULL)
        return false;

    // Try to transform the array of pointers to a single base pointer
    // and an array of int32 offsets.  (All the hard work is done by
    // lGetBasePtrAndOffsets).
    llvm::Value *ptrs = callInst->getArgOperand(0);
    llvm::Value *offsetVector = NULL;
    llvm::Value *basePtr = lGetBasePtrAndOffsets(ptrs, &offsetVector, callInst);

    if (basePtr == NULL || offsetVector == NULL ||
        (info->isGather == false && info->isPrefetch == true && g->target->hasVecPrefetch() == false)) {
        // It's actually a fully general gather/scatter with a varying
        // set of base pointers, so leave it as is and continune onward
        // to the next instruction...
        return false;
    }
    // Cast the base pointer to a void *, since that's what the
    // __pseudo_*_base_offsets_* functions want.
    basePtr = new llvm::IntToPtrInst(basePtr, LLVMTypes::VoidPointerType, llvm::Twine(basePtr->getName()) + "_2void",
                                     callInst);
    lCopyMetadata(basePtr, callInst);
    llvm::Function *gatherScatterFunc = info->baseOffsetsFunc;

    if ((info->isGather == true && g->target->hasGather()) ||
        (info->isGather == false && info->isPrefetch == false && g->target->hasScatter()) ||
        (info->isGather == false && info->isPrefetch == true && g->target->hasVecPrefetch())) {

        // See if the offsets are scaled by 2, 4, or 8.  If so,
        // extract that scale factor and rewrite the offsets to remove
        // it.
        llvm::Value *offsetScale = lExtractOffsetVector248Scale(&offsetVector);

        // If we're doing 32-bit addressing on a 64-bit target, here we
        // will see if we can call one of the 32-bit variants of the pseudo
        // gather/scatter functions.
        if (g->opt.force32BitAddressing && lOffsets32BitSafe(&offsetVector, callInst)) {
            gatherScatterFunc = info->baseOffsets32Func;
        }

        if (info->isGather || info->isPrefetch) {
            llvm::Value *mask = callInst->getArgOperand(1);

            // Generate a new function call to the next pseudo gather
            // base+offsets instruction.  Note that we're passing a NULL
            // llvm::Instruction to llvm::CallInst::Create; this means that
            // the instruction isn't inserted into a basic block and that
            // way we can then call ReplaceInstWithInst().
            llvm::Instruction *newCall = lCallInst(gatherScatterFunc, basePtr, offsetScale, offsetVector, mask,
                                                   callInst->getName().str().c_str(), NULL);
            lCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        } else {
            llvm::Value *storeValue = callInst->getArgOperand(1);
            llvm::Value *mask = callInst->getArgOperand(2);

            // Generate a new function call to the next pseudo scatter
            // base+offsets instruction.  See above for why passing NULL
            // for the Instruction * is intended.
            llvm::Instruction *newCall =
                lCallInst(gatherScatterFunc, basePtr, offsetScale, offsetVector, storeValue, mask, "", NULL);
            lCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        }
    } else {
        // Try to decompose the offset vector into a compile time constant
        // component and a varying component.  The constant component is
        // passed as a separate parameter to the gather/scatter functions,
        // which in turn allows their implementations to end up emitting
        // x86 instructions with constant offsets encoded in them.
        llvm::Value *constOffset = NULL;
        llvm::Value *variableOffset = NULL;
        lExtractConstantOffset(offsetVector, &constOffset, &variableOffset, callInst);
        if (constOffset == NULL)
            constOffset = LLVMIntAsType(0, offsetVector->getType());
        if (variableOffset == NULL)
            variableOffset = LLVMIntAsType(0, offsetVector->getType());

        // See if the varying component is scaled by 2, 4, or 8.  If so,
        // extract that scale factor and rewrite variableOffset to remove
        // it.  (This also is pulled out so that we can match the scales by
        // 2/4/8 offered by x86 addressing operators.)
        llvm::Value *offsetScale = lExtractOffsetVector248Scale(&variableOffset);

        // If we're doing 32-bit addressing on a 64-bit target, here we
        // will see if we can call one of the 32-bit variants of the pseudo
        // gather/scatter functions.
        if (g->opt.force32BitAddressing && lOffsets32BitSafe(&variableOffset, &constOffset, callInst)) {
            gatherScatterFunc = info->baseOffsets32Func;
        }

        if (info->isGather || info->isPrefetch) {
            llvm::Value *mask = callInst->getArgOperand(1);

            // Generate a new function call to the next pseudo gather
            // base+offsets instruction.  Note that we're passing a NULL
            // llvm::Instruction to llvm::CallInst::Create; this means that
            // the instruction isn't inserted into a basic block and that
            // way we can then call ReplaceInstWithInst().
            llvm::Instruction *newCall = lCallInst(gatherScatterFunc, basePtr, variableOffset, offsetScale, constOffset,
                                                   mask, callInst->getName().str().c_str(), NULL);
            lCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        } else {
            llvm::Value *storeValue = callInst->getArgOperand(1);
            llvm::Value *mask = callInst->getArgOperand(2);

            // Generate a new function call to the next pseudo scatter
            // base+offsets instruction.  See above for why passing NULL
            // for the Instruction * is intended.
            llvm::Instruction *newCall = lCallInst(gatherScatterFunc, basePtr, variableOffset, offsetScale, constOffset,
                                                   storeValue, mask, "", NULL);
            lCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        }
    }
    return true;
}

/** Try to improve the decomposition between compile-time constant and
    compile-time unknown offsets in calls to the __pseudo_*_base_offsets*
    functions.  Other other optimizations have run, we will sometimes be
    able to pull more terms out of the unknown part and add them into the
    compile-time-known part.
 */
static bool lGSBaseOffsetsGetMoreConst(llvm::CallInst *callInst) {
    struct GSBOInfo {
        GSBOInfo(const char *pgboFuncName, const char *pgbo32FuncName, bool ig, bool ip)
            : isGather(ig), isPrefetch(ip) {
            baseOffsetsFunc = m->module->getFunction(pgboFuncName);
            baseOffsets32Func = m->module->getFunction(pgbo32FuncName);
        }
        llvm::Function *baseOffsetsFunc, *baseOffsets32Func;
        const bool isGather;
        const bool isPrefetch;
    };

    GSBOInfo gsFuncs[] = {
        GSBOInfo(
            g->target->hasGather() ? "__pseudo_gather_base_offsets32_i8" : "__pseudo_gather_factored_base_offsets32_i8",
            g->target->hasGather() ? "__pseudo_gather_base_offsets32_i8" : "__pseudo_gather_factored_base_offsets32_i8",
            true, false),
        GSBOInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_i16"
                                        : "__pseudo_gather_factored_base_offsets32_i16",
                 g->target->hasGather() ? "__pseudo_gather_base_offsets32_i16"
                                        : "__pseudo_gather_factored_base_offsets32_i16",
                 true, false),
        GSBOInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_half"
                                        : "__pseudo_gather_factored_base_offsets32_half",
                 g->target->hasGather() ? "__pseudo_gather_base_offsets32_half"
                                        : "__pseudo_gather_factored_base_offsets32_half",
                 true, false),
        GSBOInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_i32"
                                        : "__pseudo_gather_factored_base_offsets32_i32",
                 g->target->hasGather() ? "__pseudo_gather_base_offsets32_i32"
                                        : "__pseudo_gather_factored_base_offsets32_i32",
                 true, false),
        GSBOInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_float"
                                        : "__pseudo_gather_factored_base_offsets32_float",
                 g->target->hasGather() ? "__pseudo_gather_base_offsets32_float"
                                        : "__pseudo_gather_factored_base_offsets32_float",
                 true, false),
        GSBOInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_i64"
                                        : "__pseudo_gather_factored_base_offsets32_i64",
                 g->target->hasGather() ? "__pseudo_gather_base_offsets32_i64"
                                        : "__pseudo_gather_factored_base_offsets32_i64",
                 true, false),
        GSBOInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_double"
                                        : "__pseudo_gather_factored_base_offsets32_double",
                 g->target->hasGather() ? "__pseudo_gather_base_offsets32_double"
                                        : "__pseudo_gather_factored_base_offsets32_double",
                 true, false),

        GSBOInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i8"
                                         : "__pseudo_scatter_factored_base_offsets32_i8",
                 g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i8"
                                         : "__pseudo_scatter_factored_base_offsets32_i8",
                 false, false),
        GSBOInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i16"
                                         : "__pseudo_scatter_factored_base_offsets32_i16",
                 g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i16"
                                         : "__pseudo_scatter_factored_base_offsets32_i16",
                 false, false),
        GSBOInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_half"
                                         : "__pseudo_scatter_factored_base_offsets32_half",
                 g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i16"
                                         : "__pseudo_scatter_factored_base_offsets32_half",
                 false, false),
        GSBOInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i32"
                                         : "__pseudo_scatter_factored_base_offsets32_i32",
                 g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i32"
                                         : "__pseudo_scatter_factored_base_offsets32_i32",
                 false, false),
        GSBOInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_float"
                                         : "__pseudo_scatter_factored_base_offsets32_float",
                 g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_float"
                                         : "__pseudo_scatter_factored_base_offsets32_float",
                 false, false),
        GSBOInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i64"
                                         : "__pseudo_scatter_factored_base_offsets32_i64",
                 g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i64"
                                         : "__pseudo_scatter_factored_base_offsets32_i64",
                 false, false),
        GSBOInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_double"
                                         : "__pseudo_scatter_factored_base_offsets32_double",
                 g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_double"
                                         : "__pseudo_scatter_factored_base_offsets32_double",
                 false, false),

        GSBOInfo(g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_1_native" : "__prefetch_read_varying_1",
                 g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_1_native" : "__prefetch_read_varying_1",
                 false, true),

        GSBOInfo(g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_2_native" : "__prefetch_read_varying_2",
                 g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_2_native" : "__prefetch_read_varying_2",
                 false, true),

        GSBOInfo(g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_3_native" : "__prefetch_read_varying_3",
                 g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_3_native" : "__prefetch_read_varying_3",
                 false, true),

        GSBOInfo(
            g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_nt_native" : "__prefetch_read_varying_nt",
            g->target->hasVecPrefetch() ? "__pseudo_prefetch_read_varying_nt_native" : "__prefetch_read_varying_nt",
            false, true),

        GSBOInfo(
            g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_1_native" : "__prefetch_write_varying_1",
            g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_1_native" : "__prefetch_write_varying_1",
            false, true),

        GSBOInfo(
            g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_2_native" : "__prefetch_write_varying_2",
            g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_2_native" : "__prefetch_write_varying_2",
            false, true),

        GSBOInfo(
            g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_3_native" : "__prefetch_write_varying_3",
            g->target->hasVecPrefetch() ? "__pseudo_prefetch_write_varying_3_native" : "__prefetch_write_varying_3",
            false, true),
    };

    int numGSFuncs = sizeof(gsFuncs) / sizeof(gsFuncs[0]);
    for (int i = 0; i < numGSFuncs; ++i)
        Assert(gsFuncs[i].baseOffsetsFunc != NULL && gsFuncs[i].baseOffsets32Func != NULL);

    llvm::Function *calledFunc = callInst->getCalledFunction();
    Assert(calledFunc != NULL);

    // Is one of the gather/scatter functins that decompose into
    // base+offsets being called?
    GSBOInfo *info = NULL;
    for (int i = 0; i < numGSFuncs; ++i)
        if (calledFunc == gsFuncs[i].baseOffsetsFunc || calledFunc == gsFuncs[i].baseOffsets32Func) {
            info = &gsFuncs[i];
            break;
        }
    if (info == NULL)
        return false;

    // Grab the old variable offset
    llvm::Value *origVariableOffset = callInst->getArgOperand(1);

    // If it's zero, we're done.  Don't go and think that we're clever by
    // adding these zeros to the constant offsets.
    if (llvm::isa<llvm::ConstantAggregateZero>(origVariableOffset))
        return false;

    // Try to decompose the old variable offset
    llvm::Value *constOffset = NULL;
    llvm::Value *variableOffset = NULL;
    lExtractConstantOffset(origVariableOffset, &constOffset, &variableOffset, callInst);

    // No luck
    if (constOffset == NULL)
        return false;

    // Total luck: everything could be moved to the constant offset
    if (variableOffset == NULL)
        variableOffset = LLVMIntAsType(0, origVariableOffset->getType());

    // We need to scale the value we add to the constant offset by the
    // 2/4/8 scale for the variable offset, if present.
    llvm::ConstantInt *varScale = llvm::dyn_cast<llvm::ConstantInt>(callInst->getArgOperand(2));
    Assert(varScale != NULL);

    llvm::Value *scaleSmear;
    if (origVariableOffset->getType() == LLVMTypes::Int64VectorType)
        scaleSmear = LLVMInt64Vector((int64_t)varScale->getZExtValue());
    else
        scaleSmear = LLVMInt32Vector((int32_t)varScale->getZExtValue());

    constOffset =
        llvm::BinaryOperator::Create(llvm::Instruction::Mul, constOffset, scaleSmear, constOffset->getName(), callInst);

    // And add the additional offset to the original constant offset
    constOffset = llvm::BinaryOperator::Create(llvm::Instruction::Add, constOffset, callInst->getArgOperand(3),
                                               callInst->getArgOperand(3)->getName(), callInst);

    // Finally, update the values of the operands to the gather/scatter
    // function.
    callInst->setArgOperand(1, variableOffset);
    callInst->setArgOperand(3, constOffset);

    return true;
}

static llvm::Value *lComputeCommonPointer(llvm::Value *base, llvm::Value *offsets, llvm::Instruction *insertBefore,
                                          int typeScale = 1) {
    llvm::Value *firstOffset = LLVMExtractFirstVectorElement(offsets);
    Assert(firstOffset != NULL);
    llvm::Value *typeScaleValue =
        firstOffset->getType() == LLVMTypes::Int32Type ? LLVMInt32(typeScale) : LLVMInt64(typeScale);
    if (g->target->isXeTarget() && typeScale > 1) {
        firstOffset = llvm::BinaryOperator::Create(llvm::Instruction::SDiv, firstOffset, typeScaleValue,
                                                   "scaled_offset", insertBefore);
    }

    return lGEPInst(base, firstOffset, "ptr", insertBefore);
}

static llvm::Constant *lGetOffsetScaleVec(llvm::Value *offsetScale, llvm::Type *vecType) {
    llvm::ConstantInt *offsetScaleInt = llvm::dyn_cast<llvm::ConstantInt>(offsetScale);
    Assert(offsetScaleInt != NULL);
    uint64_t scaleValue = offsetScaleInt->getZExtValue();

    std::vector<llvm::Constant *> scales;
    for (int i = 0; i < g->target->getVectorWidth(); ++i) {
        if (vecType == LLVMTypes::Int64VectorType)
            scales.push_back(LLVMInt64(scaleValue));
        else {
            Assert(vecType == LLVMTypes::Int32VectorType);
            scales.push_back(LLVMInt32((int32_t)scaleValue));
        }
    }
    return llvm::ConstantVector::get(scales);
}

/** After earlier optimization passes have run, we are sometimes able to
    determine that gathers/scatters are actually accessing memory in a more
    regular fashion and then change the operation to something simpler and
    more efficient.  For example, if all of the lanes in a gather are
    reading from the same location, we can instead do a scalar load and
    broadcast.  This pass examines gathers and scatters and tries to
    simplify them if at all possible.

    @todo Currently, this only looks for all program instances going to the
    same location and all going to a linear sequence of locations in
    memory.  There are a number of other cases that might make sense to
    look for, including things that could be handled with a vector load +
    shuffle or things that could be handled with hybrids of e.g. 2 4-wide
    vector loads with AVX, etc.
*/
static bool lGSToLoadStore(llvm::CallInst *callInst) {
    struct GatherImpInfo {
        GatherImpInfo(const char *pName, const char *lmName, const char *bmName, llvm::Type *st, int a)
            : align(a), isFactored(!g->target->hasGather()) {
            pseudoFunc = m->module->getFunction(pName);
            loadMaskedFunc = m->module->getFunction(lmName);
            blendMaskedFunc = m->module->getFunction(bmName);
            Assert(pseudoFunc != NULL && loadMaskedFunc != NULL);
            scalarType = st;
        }

        llvm::Function *pseudoFunc;
        llvm::Function *loadMaskedFunc;
        llvm::Function *blendMaskedFunc;
        llvm::Type *scalarType;
        const int align;
        const bool isFactored;
    };

    GatherImpInfo gInfo[] = {
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_i8"
                                             : "__pseudo_gather_factored_base_offsets32_i8",
                      "__masked_load_i8", "__masked_load_blend_i8", LLVMTypes::Int8Type, 1),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_i16"
                                             : "__pseudo_gather_factored_base_offsets32_i16",
                      "__masked_load_i16", "__masked_load_blend_i16", LLVMTypes::Int16Type, 2),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_half"
                                             : "__pseudo_gather_factored_base_offsets32_half",
                      "__masked_load_half", "__masked_load_blend_half", LLVMTypes::Float16Type, 2),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_i32"
                                             : "__pseudo_gather_factored_base_offsets32_i32",
                      "__masked_load_i32", "__masked_load_blend_i32", LLVMTypes::Int32Type, 4),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_float"
                                             : "__pseudo_gather_factored_base_offsets32_float",
                      "__masked_load_float", "__masked_load_blend_float", LLVMTypes::FloatType, 4),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_i64"
                                             : "__pseudo_gather_factored_base_offsets32_i64",
                      "__masked_load_i64", "__masked_load_blend_i64", LLVMTypes::Int64Type, 8),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets32_double"
                                             : "__pseudo_gather_factored_base_offsets32_double",
                      "__masked_load_double", "__masked_load_blend_double", LLVMTypes::DoubleType, 8),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets64_i8"
                                             : "__pseudo_gather_factored_base_offsets64_i8",
                      "__masked_load_i8", "__masked_load_blend_i8", LLVMTypes::Int8Type, 1),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets64_i16"
                                             : "__pseudo_gather_factored_base_offsets64_i16",
                      "__masked_load_i16", "__masked_load_blend_i16", LLVMTypes::Int16Type, 2),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets64_half"
                                             : "__pseudo_gather_factored_base_offsets64_half",
                      "__masked_load_half", "__masked_load_blend_half", LLVMTypes::Float16Type, 2),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets64_i32"
                                             : "__pseudo_gather_factored_base_offsets64_i32",
                      "__masked_load_i32", "__masked_load_blend_i32", LLVMTypes::Int32Type, 4),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets64_float"
                                             : "__pseudo_gather_factored_base_offsets64_float",
                      "__masked_load_float", "__masked_load_blend_float", LLVMTypes::FloatType, 4),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets64_i64"
                                             : "__pseudo_gather_factored_base_offsets64_i64",
                      "__masked_load_i64", "__masked_load_blend_i64", LLVMTypes::Int64Type, 8),
        GatherImpInfo(g->target->hasGather() ? "__pseudo_gather_base_offsets64_double"
                                             : "__pseudo_gather_factored_base_offsets64_double",
                      "__masked_load_double", "__masked_load_blend_double", LLVMTypes::DoubleType, 8),
    };

    struct ScatterImpInfo {
        ScatterImpInfo(const char *pName, const char *msName, llvm::Type *vpt, int a)
            : align(a), isFactored(!g->target->hasScatter()) {
            pseudoFunc = m->module->getFunction(pName);
            maskedStoreFunc = m->module->getFunction(msName);
            vecPtrType = vpt;
            Assert(pseudoFunc != NULL && maskedStoreFunc != NULL);
        }
        llvm::Function *pseudoFunc;
        llvm::Function *maskedStoreFunc;
        llvm::Type *vecPtrType;
        const int align;
        const bool isFactored;
    };

    ScatterImpInfo sInfo[] = {
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i8"
                                               : "__pseudo_scatter_factored_base_offsets32_i8",
                       "__pseudo_masked_store_i8", LLVMTypes::Int8VectorPointerType, 1),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i16"
                                               : "__pseudo_scatter_factored_base_offsets32_i16",
                       "__pseudo_masked_store_i16", LLVMTypes::Int16VectorPointerType, 2),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_half"
                                               : "__pseudo_scatter_factored_base_offsets32_half",
                       "__pseudo_masked_store_half", LLVMTypes::Float16VectorPointerType, 2),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i32"
                                               : "__pseudo_scatter_factored_base_offsets32_i32",
                       "__pseudo_masked_store_i32", LLVMTypes::Int32VectorPointerType, 4),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_float"
                                               : "__pseudo_scatter_factored_base_offsets32_float",
                       "__pseudo_masked_store_float", LLVMTypes::FloatVectorPointerType, 4),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_i64"
                                               : "__pseudo_scatter_factored_base_offsets32_i64",
                       "__pseudo_masked_store_i64", LLVMTypes::Int64VectorPointerType, 8),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets32_double"
                                               : "__pseudo_scatter_factored_base_offsets32_double",
                       "__pseudo_masked_store_double", LLVMTypes::DoubleVectorPointerType, 8),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_i8"
                                               : "__pseudo_scatter_factored_base_offsets64_i8",
                       "__pseudo_masked_store_i8", LLVMTypes::Int8VectorPointerType, 1),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_i16"
                                               : "__pseudo_scatter_factored_base_offsets64_i16",
                       "__pseudo_masked_store_i16", LLVMTypes::Int16VectorPointerType, 2),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_half"
                                               : "__pseudo_scatter_factored_base_offsets64_half",
                       "__pseudo_masked_store_half", LLVMTypes::Float16VectorPointerType, 2),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_i32"
                                               : "__pseudo_scatter_factored_base_offsets64_i32",
                       "__pseudo_masked_store_i32", LLVMTypes::Int32VectorPointerType, 4),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_float"
                                               : "__pseudo_scatter_factored_base_offsets64_float",
                       "__pseudo_masked_store_float", LLVMTypes::FloatVectorPointerType, 4),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_i64"
                                               : "__pseudo_scatter_factored_base_offsets64_i64",
                       "__pseudo_masked_store_i64", LLVMTypes::Int64VectorPointerType, 8),
        ScatterImpInfo(g->target->hasScatter() ? "__pseudo_scatter_base_offsets64_double"
                                               : "__pseudo_scatter_factored_base_offsets64_double",
                       "__pseudo_masked_store_double", LLVMTypes::DoubleVectorPointerType, 8),
    };

    llvm::Function *calledFunc = callInst->getCalledFunction();

    GatherImpInfo *gatherInfo = NULL;
    ScatterImpInfo *scatterInfo = NULL;
    for (unsigned int i = 0; i < sizeof(gInfo) / sizeof(gInfo[0]); ++i) {
        if (gInfo[i].pseudoFunc != NULL && calledFunc == gInfo[i].pseudoFunc) {
            gatherInfo = &gInfo[i];
            break;
        }
    }
    for (unsigned int i = 0; i < sizeof(sInfo) / sizeof(sInfo[0]); ++i) {
        if (sInfo[i].pseudoFunc != NULL && calledFunc == sInfo[i].pseudoFunc) {
            scatterInfo = &sInfo[i];
            break;
        }
    }
    if (gatherInfo == NULL && scatterInfo == NULL)
        return false;

    SourcePos pos;
    lGetSourcePosFromMetadata(callInst, &pos);

    llvm::Value *base = callInst->getArgOperand(0);
    llvm::Value *fullOffsets = NULL;
    llvm::Value *storeValue = NULL;
    llvm::Value *mask = NULL;
    if ((gatherInfo != NULL && gatherInfo->isFactored) || (scatterInfo != NULL && scatterInfo->isFactored)) {
        llvm::Value *varyingOffsets = callInst->getArgOperand(1);
        llvm::Value *offsetScale = callInst->getArgOperand(2);
        llvm::Value *constOffsets = callInst->getArgOperand(3);
        if (scatterInfo)
            storeValue = callInst->getArgOperand(4);
        mask = callInst->getArgOperand((gatherInfo != NULL) ? 4 : 5);

        // Compute the full offset vector: offsetScale * varyingOffsets + constOffsets
        llvm::Constant *offsetScaleVec = lGetOffsetScaleVec(offsetScale, varyingOffsets->getType());

        llvm::Value *scaledVarying = llvm::BinaryOperator::Create(llvm::Instruction::Mul, offsetScaleVec,
                                                                  varyingOffsets, "scaled_varying", callInst);
        fullOffsets = llvm::BinaryOperator::Create(llvm::Instruction::Add, scaledVarying, constOffsets,
                                                   "varying+const_offsets", callInst);
    } else {
        if (scatterInfo)
            storeValue = callInst->getArgOperand(3);
        mask = callInst->getArgOperand((gatherInfo != NULL) ? 3 : 4);

        llvm::Value *offsetScale = callInst->getArgOperand(1);
        llvm::Value *offsets = callInst->getArgOperand(2);
        llvm::Value *offsetScaleVec = lGetOffsetScaleVec(offsetScale, offsets->getType());

        fullOffsets =
            llvm::BinaryOperator::Create(llvm::Instruction::Mul, offsetScaleVec, offsets, "scaled_offsets", callInst);
    }

    Debug(SourcePos(), "GSToLoadStore: %s.", fullOffsets->getName().str().c_str());
    llvm::Type *scalarType = (gatherInfo != NULL) ? gatherInfo->scalarType : scatterInfo->vecPtrType->getScalarType();
    int typeScale = g->target->getDataLayout()->getTypeStoreSize(scalarType) /
                    g->target->getDataLayout()->getTypeStoreSize(base->getType()->getContainedType(0));

    if (LLVMVectorValuesAllEqual(fullOffsets)) {
        // If all the offsets are equal, then compute the single
        // pointer they all represent based on the first one of them
        // (arbitrarily).
        if (gatherInfo != NULL) {
            // A gather with everyone going to the same location is
            // handled as a scalar load and broadcast across the lanes.
            Debug(pos, "Transformed gather to scalar load and broadcast!");
            llvm::Value *ptr;
            // For Xe we need to cast the base first and only after that get common pointer otherwise
            // CM backend will be broken on bitcast i8* to T* instruction with following load.
            // For this we need to re-calculate the offset basing on type sizes.
            if (g->target->isXeTarget()) {
                base = new llvm::BitCastInst(base, llvm::PointerType::get(scalarType, 0), base->getName(), callInst);
                ptr = lComputeCommonPointer(base, fullOffsets, callInst, typeScale);
            } else {
                ptr = lComputeCommonPointer(base, fullOffsets, callInst);
                ptr = new llvm::BitCastInst(ptr, llvm::PointerType::get(scalarType, 0), base->getName(), callInst);
            }

            lCopyMetadata(ptr, callInst);
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
            Assert(llvm::isa<llvm::PointerType>(ptr->getType()));
            llvm::Value *scalarValue =
                new llvm::LoadInst(llvm::dyn_cast<llvm::PointerType>(ptr->getType())->getPointerElementType(), ptr,
                                   callInst->getName(), callInst);
#else
            llvm::Value *scalarValue = new llvm::LoadInst(ptr, callInst->getName(), callInst);
#endif

            // Generate the following sequence:
            //   %name123 = insertelement <4 x i32> undef, i32 %val, i32 0
            //   %name124 = shufflevector <4 x i32> %name123, <4 x i32> undef,
            //                                              <4 x i32> zeroinitializer
            llvm::Value *undef1Value = llvm::UndefValue::get(callInst->getType());
            llvm::Value *undef2Value = llvm::UndefValue::get(callInst->getType());
            llvm::Value *insertVec =
                llvm::InsertElementInst::Create(undef1Value, scalarValue, LLVMInt32(0), callInst->getName(), callInst);
            llvm::Value *zeroMask =
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
                llvm::ConstantVector::getSplat(callInst->getType()->getVectorNumElements(),
#elif ISPC_LLVM_VERSION < ISPC_LLVM_12_0
                llvm::ConstantVector::getSplat(
                    {llvm::dyn_cast<llvm::VectorType>(callInst->getType())->getNumElements(), false},

#else // LLVM 12.0+
                llvm::ConstantVector::getSplat(
                    llvm::ElementCount::get(
                        llvm::dyn_cast<llvm::FixedVectorType>(callInst->getType())->getNumElements(), false),
#endif
                                               llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*g->ctx)));
            llvm::Value *shufValue = new llvm::ShuffleVectorInst(insertVec, undef2Value, zeroMask, callInst->getName());

            lCopyMetadata(shufValue, callInst);
            llvm::ReplaceInstWithInst(callInst, llvm::dyn_cast<llvm::Instruction>(shufValue));
            return true;
        } else {
            // A scatter with everyone going to the same location is
            // undefined (if there's more than one program instance in
            // the gang).  Issue a warning.
            if (g->target->getVectorWidth() > 1)
                Warning(pos, "Undefined behavior: all program instances are "
                             "writing to the same location!");

            // We could do something similar to the gather case, where
            // we arbitrarily write one of the values, but we need to
            // a) check to be sure the mask isn't all off and b) pick
            // the value from an executing program instance in that
            // case.  We'll just let a bunch of the program instances
            // do redundant writes, since this isn't important to make
            // fast anyway...
            return false;
        }
    } else {
        int step = gatherInfo ? gatherInfo->align : scatterInfo->align;
        if (step > 0 && LLVMVectorIsLinear(fullOffsets, step)) {
            // We have a linear sequence of memory locations being accessed
            // starting with the location given by the offset from
            // offsetElements[0], with stride of 4 or 8 bytes (for 32 bit
            // and 64 bit gather/scatters, respectively.
            llvm::Value *ptr;

            if (gatherInfo != NULL) {
                if (g->target->isXeTarget()) {
                    // For Xe we need to cast the base first and only after that get common pointer otherwise
                    // CM backend will be broken on bitcast i8* to T* instruction with following load.
                    // For this we need to re-calculate the offset basing on type sizes.
                    // Second bitcast to void* does not cause such problem in backend.
                    base =
                        new llvm::BitCastInst(base, llvm::PointerType::get(scalarType, 0), base->getName(), callInst);
                    ptr = lComputeCommonPointer(base, fullOffsets, callInst, typeScale);
                    ptr = new llvm::BitCastInst(ptr, LLVMTypes::Int8PointerType, base->getName(), callInst);
                } else {
                    ptr = lComputeCommonPointer(base, fullOffsets, callInst);
                }
                lCopyMetadata(ptr, callInst);
                Debug(pos, "Transformed gather to unaligned vector load!");
                bool doBlendLoad = false;
#ifdef ISPC_XE_ENABLED
                doBlendLoad = g->target->isXeTarget() && g->opt.enableXeUnsafeMaskedLoad;
#endif
                llvm::Instruction *newCall =
                    lCallInst(doBlendLoad ? gatherInfo->blendMaskedFunc : gatherInfo->loadMaskedFunc, ptr, mask,
                              llvm::Twine(ptr->getName()) + "_masked_load");
                lCopyMetadata(newCall, callInst);
                llvm::ReplaceInstWithInst(callInst, newCall);
                return true;
            } else {
                Debug(pos, "Transformed scatter to unaligned vector store!");
                ptr = lComputeCommonPointer(base, fullOffsets, callInst);
                ptr = new llvm::BitCastInst(ptr, scatterInfo->vecPtrType, "ptrcast", callInst);
                llvm::Instruction *newCall = lCallInst(scatterInfo->maskedStoreFunc, ptr, storeValue, mask, "");
                lCopyMetadata(newCall, callInst);
                llvm::ReplaceInstWithInst(callInst, newCall);
                return true;
            }
        }
        return false;
    }
}

///////////////////////////////////////////////////////////////////////////
// MaskedStoreOptPass

#ifdef ISPC_XE_ENABLED
static llvm::Function *lXeMaskedInst(llvm::Instruction *inst, bool isStore, llvm::Type *type) {
    std::string maskedFuncName;
    if (isStore) {
        maskedFuncName = "masked_store_";
    } else {
        maskedFuncName = "masked_load_";
    }
    if (type == LLVMTypes::Int8Type)
        maskedFuncName += "i8";
    else if (type == LLVMTypes::Int16Type)
        maskedFuncName += "i16";
    else if (type == LLVMTypes::Int32Type)
        maskedFuncName += "i32";
    else if (type == LLVMTypes::Int64Type)
        maskedFuncName += "i64";
    else if (type == LLVMTypes::Float16Type)
        maskedFuncName += "half";
    else if (type == LLVMTypes::FloatType)
        maskedFuncName += "float";
    else if (type == LLVMTypes::DoubleType)
        maskedFuncName += "double";

    llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(inst);
    if (callInst != NULL && callInst->getCalledFunction() != NULL &&
        callInst->getCalledFunction()->getName().contains(maskedFuncName)) {
        return NULL;
    }
    return m->module->getFunction("__" + maskedFuncName);
}

static llvm::CallInst *lXeStoreInst(llvm::Value *val, llvm::Value *ptr, llvm::Instruction *inst) {
    Assert(g->target->isXeTarget());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    Assert(llvm::isa<llvm::FixedVectorType>(val->getType()));
    llvm::FixedVectorType *valVecType = llvm::dyn_cast<llvm::FixedVectorType>(val->getType());
#else
    Assert(llvm::isa<llvm::VectorType>(val->getType()));
    llvm::VectorType *valVecType = llvm::dyn_cast<llvm::VectorType>(val->getType());
#endif
    Assert(llvm::isPowerOf2_32(valVecType->getNumElements()));

    // The data write of svm store must have a size that is a power of two from 16 to 128
    // bytes. However for int8 type and simd width = 8, the data write size is 8.
    // So we use masked store function here instead of svm store which process int8 type
    // correctly.
    bool isMaskedStoreRequired = false;
    if (valVecType->getPrimitiveSizeInBits() / 8 < 16) {
        Assert(valVecType->getScalarType() == LLVMTypes::Int8Type && g->target->getVectorWidth() == 8);
        isMaskedStoreRequired = true;
    } else if (valVecType->getPrimitiveSizeInBits() / 8 > 8 * OWORD) {
        // The data write of svm store must be less than 8 * OWORD. However for
        // double or int64 types for simd32 targets it is bigger so use masked_store implementation
        Assert((valVecType->getScalarType() == LLVMTypes::Int64Type ||
                valVecType->getScalarType() == LLVMTypes::DoubleType) &&
               g->target->getVectorWidth() == 32);
        isMaskedStoreRequired = true;
    }
    if (isMaskedStoreRequired) {
        if (llvm::Function *maskedFunc = lXeMaskedInst(inst, true, valVecType->getScalarType())) {
            return llvm::dyn_cast<llvm::CallInst>(lCallInst(maskedFunc, ptr, val, LLVMMaskAllOn, ""));
        } else {
            return NULL;
        }
    }

    llvm::Instruction *svm_st_zext = new llvm::PtrToIntInst(ptr, LLVMTypes::Int64Type, "svm_st_ptrtoint", inst);

    llvm::Type *argTypes[] = {svm_st_zext->getType(), val->getType()};
    auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(m->module, llvm::GenXIntrinsic::genx_svm_block_st, argTypes);
    return llvm::CallInst::Create(Fn, {svm_st_zext, val}, inst->getName());
}

static llvm::CallInst *lXeLoadInst(llvm::Value *ptr, llvm::Type *retType, llvm::Instruction *inst) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    Assert(llvm::isa<llvm::FixedVectorType>(retType));
    llvm::FixedVectorType *retVecType = llvm::dyn_cast<llvm::FixedVectorType>(retType);
#else
    Assert(llvm::isa<llvm::VectorType>(retType));
    llvm::VectorType *retVecType = llvm::dyn_cast<llvm::VectorType>(retType);
#endif
    Assert(llvm::isPowerOf2_32(retVecType->getNumElements()));
    Assert(retVecType->getPrimitiveSizeInBits());
    // The data read of svm load must have a size that is a power of two from 16 to 128
    // bytes. However for int8 type and simd width = 8, the data read size is 8.
    // So we use masked load function here instead of svm load which process int8 type
    // correctly.
    bool isMaskedLoadRequired = false;
    if (retVecType->getPrimitiveSizeInBits() / 8 < 16) {
        Assert(retVecType->getScalarType() == LLVMTypes::Int8Type && g->target->getVectorWidth() == 8);
        isMaskedLoadRequired = true;
    } else if (retVecType->getPrimitiveSizeInBits() / 8 > 8 * OWORD) {
        // The data write of svm store must be less than 8 * OWORD. However for
        // double or int64 types for simd32 targets it is bigger so use masked_store implementation
        Assert((retVecType->getScalarType() == LLVMTypes::Int64Type ||
                retVecType->getScalarType() == LLVMTypes::DoubleType) &&
               g->target->getVectorWidth() == 32);
        isMaskedLoadRequired = true;
    }

    if (isMaskedLoadRequired) {
        if (llvm::Function *maskedFunc = lXeMaskedInst(inst, false, retVecType->getScalarType())) {
            // <WIDTH x $1> @__masked_load_i8(i8 *, <WIDTH x MASK> %mask)
            // Cast pointer to i8*
            ptr = new llvm::BitCastInst(ptr, LLVMTypes::Int8PointerType, "ptr_to_i8", inst);
            return llvm::dyn_cast<llvm::CallInst>(lCallInst(maskedFunc, ptr, LLVMMaskAllOn, "_masked_load_"));
        } else {
            return NULL;
        }
    }
    llvm::Value *svm_ld_ptrtoint = new llvm::PtrToIntInst(ptr, LLVMTypes::Int64Type, "svm_ld_ptrtoint", inst);

    auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(m->module, llvm::GenXIntrinsic::genx_svm_block_ld_unaligned,
                                                      {retType, svm_ld_ptrtoint->getType()});

    return llvm::CallInst::Create(Fn, svm_ld_ptrtoint, inst->getName());
}
#endif
/** Masked stores are generally more complex than regular stores; for
    example, they require multiple instructions to simulate under SSE.
    This optimization detects cases where masked stores can be replaced
    with regular stores or removed entirely, for the cases of an 'all on'
    mask and an 'all off' mask, respectively.
*/
static bool lImproveMaskedStore(llvm::CallInst *callInst) {
    struct MSInfo {
        MSInfo(const char *name, const int a) : align(a) {
            func = m->module->getFunction(name);
            Assert(func != NULL);
        }
        llvm::Function *func;
        const int align;
    };

    MSInfo msInfo[] = {MSInfo("__pseudo_masked_store_i8", 1),
                       MSInfo("__pseudo_masked_store_i16", 2),
                       MSInfo("__pseudo_masked_store_half", 2),
                       MSInfo("__pseudo_masked_store_i32", 4),
                       MSInfo("__pseudo_masked_store_float", 4),
                       MSInfo("__pseudo_masked_store_i64", 8),
                       MSInfo("__pseudo_masked_store_double", 8),
                       MSInfo("__masked_store_blend_i8", 1),
                       MSInfo("__masked_store_blend_i16", 2),
                       MSInfo("__masked_store_blend_half", 2),
                       MSInfo("__masked_store_blend_i32", 4),
                       MSInfo("__masked_store_blend_float", 4),
                       MSInfo("__masked_store_blend_i64", 8),
                       MSInfo("__masked_store_blend_double", 8),
                       MSInfo("__masked_store_i8", 1),
                       MSInfo("__masked_store_i16", 2),
                       MSInfo("__masked_store_half", 2),
                       MSInfo("__masked_store_i32", 4),
                       MSInfo("__masked_store_float", 4),
                       MSInfo("__masked_store_i64", 8),
                       MSInfo("__masked_store_double", 8)};
    llvm::Function *called = callInst->getCalledFunction();

    int nMSFuncs = sizeof(msInfo) / sizeof(msInfo[0]);
    MSInfo *info = NULL;
    for (int i = 0; i < nMSFuncs; ++i) {
        if (msInfo[i].func != NULL && called == msInfo[i].func) {
            info = &msInfo[i];
            break;
        }
    }
    if (info == NULL)
        return false;

    // Got one; grab the operands
    llvm::Value *lvalue = callInst->getArgOperand(0);
    llvm::Value *rvalue = callInst->getArgOperand(1);
    llvm::Value *mask = callInst->getArgOperand(2);

    MaskStatus maskStatus = lGetMaskStatus(mask);
    if (maskStatus == MaskStatus::all_off) {
        // Zero mask - no-op, so remove the store completely.  (This
        // may in turn lead to being able to optimize out instructions
        // that compute the rvalue...)
        callInst->eraseFromParent();
        return true;
    } else if (maskStatus == MaskStatus::all_on) {
        // The mask is all on, so turn this into a regular store
        llvm::Type *rvalueType = rvalue->getType();
        llvm::Instruction *store = NULL;
#ifdef ISPC_XE_ENABLED
        // InternalLinkage check is to prevent generation of SVM store when the pointer came from caller.
        // Since it can be allocated in a caller, it may be allocated on register. Possible svm store
        // is resolved after inlining. TODO: problems can be met here in case of Stack Calls.
        if (g->target->isXeTarget() && GetAddressSpace(lvalue) == AddressSpace::ispc_global &&
            callInst->getParent()->getParent()->getLinkage() != llvm::GlobalValue::LinkageTypes::InternalLinkage) {
            store = lXeStoreInst(rvalue, lvalue, callInst);
        } else if (!g->target->isXeTarget() ||
                   (g->target->isXeTarget() && GetAddressSpace(lvalue) == AddressSpace::ispc_default))
#endif
        {
            llvm::Type *ptrType = llvm::PointerType::get(rvalueType, 0);

            lvalue = new llvm::BitCastInst(lvalue, ptrType, "lvalue_to_ptr_type", callInst);
            lCopyMetadata(lvalue, callInst);
            store = new llvm::StoreInst(
                rvalue, lvalue, false /* not volatile */,
                llvm::MaybeAlign(g->opt.forceAlignedMemory ? g->target->getNativeVectorAlignment() : info->align)
                    .valueOrOne());
        }
        if (store != NULL) {
            lCopyMetadata(store, callInst);
            llvm::ReplaceInstWithInst(callInst, store);
            return true;
        }
#ifdef ISPC_XE_ENABLED
    } else {
        if (g->target->isXeTarget() && GetAddressSpace(lvalue) == AddressSpace::ispc_global) {
            // In thuis case we use masked_store which on Xe target causes scatter usage.
            // Get the source position from the metadata attached to the call
            // instruction so that we can issue PerformanceWarning()s below.
            SourcePos pos;
            bool gotPosition = lGetSourcePosFromMetadata(callInst, &pos);
            if (gotPosition) {
                PerformanceWarning(pos, "Scatter required to store value.");
            }
        }
#endif
    }
    return false;
}

static bool lImproveMaskedLoad(llvm::CallInst *callInst, llvm::BasicBlock::iterator iter) {
    struct MLInfo {
        MLInfo(const char *name, const int a) : align(a) {
            func = m->module->getFunction(name);
            Assert(func != NULL);
        }
        llvm::Function *func;
        const int align;
    };

    llvm::Function *called = callInst->getCalledFunction();
    // TODO: we should use dynamic data structure for MLInfo and fill
    // it differently for Xe and CPU targets. It will also help
    // to avoid declaration of Xe intrinsics for CPU targets.
    // It should be changed seamlessly here and in all similar places in this file.
    MLInfo mlInfo[] = {MLInfo("__masked_load_i8", 1),    MLInfo("__masked_load_i16", 2),
                       MLInfo("__masked_load_half", 2),  MLInfo("__masked_load_i32", 4),
                       MLInfo("__masked_load_float", 4), MLInfo("__masked_load_i64", 8),
                       MLInfo("__masked_load_double", 8)};
    MLInfo xeInfo[] = {MLInfo("__masked_load_i8", 1),        MLInfo("__masked_load_i16", 2),
                       MLInfo("__masked_load_half", 2),      MLInfo("__masked_load_i32", 4),
                       MLInfo("__masked_load_float", 4),     MLInfo("__masked_load_i64", 8),
                       MLInfo("__masked_load_double", 8),    MLInfo("__masked_load_blend_i8", 1),
                       MLInfo("__masked_load_blend_i16", 2), MLInfo("__masked_load_blend_half", 2),
                       MLInfo("__masked_load_blend_i32", 4), MLInfo("__masked_load_blend_float", 4),
                       MLInfo("__masked_load_blend_i64", 8), MLInfo("__masked_load_blend_double", 8)};
    MLInfo *info = NULL;
    if (g->target->isXeTarget()) {
        int nFuncs = sizeof(xeInfo) / sizeof(xeInfo[0]);
        for (int i = 0; i < nFuncs; ++i) {
            if (xeInfo[i].func != NULL && called == xeInfo[i].func) {
                info = &xeInfo[i];
                break;
            }
        }
    } else {
        int nFuncs = sizeof(mlInfo) / sizeof(mlInfo[0]);
        for (int i = 0; i < nFuncs; ++i) {
            if (mlInfo[i].func != NULL && called == mlInfo[i].func) {
                info = &mlInfo[i];
                break;
            }
        }
    }
    if (info == NULL)
        return false;

    // Got one; grab the operands
    llvm::Value *ptr = callInst->getArgOperand(0);
    llvm::Value *mask = callInst->getArgOperand(1);

    MaskStatus maskStatus = lGetMaskStatus(mask);
    if (maskStatus == MaskStatus::all_off) {
        // Zero mask - no-op, so replace the load with an undef value
        llvm::ReplaceInstWithValue(iter->getParent()->getInstList(), iter, llvm::UndefValue::get(callInst->getType()));
        return true;
    } else if (maskStatus == MaskStatus::all_on) {
        // The mask is all on, so turn this into a regular load
        llvm::Instruction *load = NULL;
#ifdef ISPC_XE_ENABLED
        // InternalLinkage check is to prevent generation of SVM load when the pointer came from caller.
        // Since it can be allocated in a caller, it may be allocated on register. Possible svm load
        // is resolved after inlining. TODO: problems can be met here in case of Stack Calls.
        if (g->target->isXeTarget() && GetAddressSpace(ptr) == AddressSpace::ispc_global &&
            callInst->getParent()->getParent()->getLinkage() != llvm::GlobalValue::LinkageTypes::InternalLinkage) {
            load = lXeLoadInst(ptr, callInst->getType(), callInst);
        } else if (!g->target->isXeTarget() ||
                   (g->target->isXeTarget() && GetAddressSpace(ptr) == AddressSpace::ispc_default))
#endif
        {
            llvm::Type *ptrType = llvm::PointerType::get(callInst->getType(), 0);
            ptr = new llvm::BitCastInst(ptr, ptrType, "ptr_cast_for_load", callInst);
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
            load = new llvm::LoadInst(
                ptr, callInst->getName(), false /* not volatile */,
                llvm::MaybeAlign(g->opt.forceAlignedMemory ? g->target->getNativeVectorAlignment() : info->align)
                    .valueOrOne(),
                (llvm::Instruction *)NULL);
#else // LLVM 11.0+
            Assert(llvm::isa<llvm::PointerType>(ptr->getType()));
            load = new llvm::LoadInst(
                llvm::dyn_cast<llvm::PointerType>(ptr->getType())->getPointerElementType(), ptr, callInst->getName(),
                false /* not volatile */,
                llvm::MaybeAlign(g->opt.forceAlignedMemory ? g->target->getNativeVectorAlignment() : info->align)
                    .valueOrOne(),
                (llvm::Instruction *)NULL);
#endif
        }
        if (load != NULL) {
            lCopyMetadata(load, callInst);
            llvm::ReplaceInstWithInst(callInst, load);
            return true;
        }
    }
    return false;
}

bool ImproveMemoryOpsPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("ImproveMemoryOps");

    bool modifiedAny = false;
restart:
    // Iterate through all of the instructions in the basic block.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        // If we don't have a call to one of the
        // __pseudo_{gather,scatter}_* functions, then just go on to the
        // next instruction.
        if (callInst == NULL || callInst->getCalledFunction() == NULL)
            continue;
        if (lGSToGSBaseOffsets(callInst)) {
            modifiedAny = true;
            goto restart;
        }
        if (lGSBaseOffsetsGetMoreConst(callInst)) {
            modifiedAny = true;
            goto restart;
        }
        if (lGSToLoadStore(callInst)) {
            modifiedAny = true;
            goto restart;
        }
        if (lImproveMaskedStore(callInst)) {
            modifiedAny = true;
            goto restart;
        }
        if (lImproveMaskedLoad(callInst, iter)) {
            modifiedAny = true;
            goto restart;
        }
    }

    DEBUG_END_PASS("ImproveMemoryOps");

    return modifiedAny;
}

bool ImproveMemoryOpsPass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("ImproveMemoryOpsPass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateImproveMemoryOpsPass() { return new ImproveMemoryOpsPass; }

///////////////////////////////////////////////////////////////////////////
// GatherCoalescePass

// This pass implements two optimizations to improve the performance of
// gathers; currently only gathers of 32-bit values where it can be
// determined at compile time that the mask is all on are supported, though
// both of those limitations may be generalized in the future.
//
//  First, for any single gather, see if it's worthwhile to break it into
//  any of scalar, 2-wide (i.e. 64-bit), 4-wide, or 8-wide loads.  Further,
//  we generate code that shuffles these loads around.  Doing fewer, larger
//  loads in this manner, when possible, can be more efficient.
//
//  Second, this pass can coalesce memory accesses across multiple
//  gathers. If we have a series of gathers without any memory writes in
//  the middle, then we try to analyze their reads collectively and choose
//  an efficient set of loads for them.  Not only does this help if
//  different gathers reuse values from the same location in memory, but
//  it's specifically helpful when data with AOS layout is being accessed;
//  in this case, we're often able to generate wide vector loads and
//  appropriate shuffles automatically.

class GatherCoalescePass : public llvm::FunctionPass {
  public:
    static char ID;
    GatherCoalescePass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Gather Coalescing"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);
};

char GatherCoalescePass::ID = 0;

/** Representation of a memory load that the gather coalescing code has
    decided to generate.
 */
struct CoalescedLoadOp {
    CoalescedLoadOp(int64_t s, int c) {
        start = s;
        count = c;
        load = element0 = element1 = NULL;
    }

    /** Starting offset of the load from the common base pointer (in terms
        of numbers of items of the underlying element type--*not* in terms
        of bytes). */
    int64_t start;

    /** Number of elements to load at this location */
    int count;

    /** Value loaded from memory for this load op */
    llvm::Value *load;

    /** For 2-wide loads (i.e. 64-bit loads), these store the lower and
        upper 32 bits of the result, respectively. */
    llvm::Value *element0, *element1;
};

/** This function determines whether it makes sense (and is safe) to
    generate a vector load of width vectorWidth, starting at *iter.  It
    returns true if so, setting *newIter to point to the next element in
    the set that isn't taken care of by the generated load.  If a vector
    load of the given width doesn't make sense, then false is returned.
 */
static bool lVectorLoadIsEfficient(std::set<int64_t>::iterator iter, std::set<int64_t>::iterator end,
                                   std::set<int64_t>::iterator *newIter, int vectorWidth) {
    // We're considering a vector load of width vectorWidth, starting at
    // the offset "start".
    int64_t start = *iter;

    // The basic idea is that we'll look at the subsequent elements in the
    // load set after the initial one at start.  As long as subsequent
    // elements:
    //
    // 1. Aren't so far separated that they no longer fit into the range
    //    [start, start+vectorWidth)
    //
    // 2. And don't have too large a gap in between them (e.g., it's not
    //    worth generating an 8-wide load for two elements with offsets 0
    //    and 7, but no loads requested in between).
    //
    // Then we continue moving forward through the elements until we either
    // fill up the vector or run out of elements.

    // lastAccepted holds the last offset we've processed and accepted as
    // valid for the vector load underconsideration
    int64_t lastAccepted = start;

    while (iter != end) {
        // What is the separation in offset values from the last element we
        // added to the set for this load?
        int64_t delta = *iter - lastAccepted;
        if (delta > 3)
            // If there's too big a gap, then we won't issue the load
            return false;

        int64_t span = *iter - start + 1;

        if (span == vectorWidth) {
            // We've extended far enough that we have exactly filled up the
            // entire vector width; we can't go any further, so return with
            // success.  (Update *newIter to point at the next element
            // after the last one accepted here.)
            *newIter = ++iter;
            return true;
        } else if (span > vectorWidth) {
            // The current offset won't fit into a vectorWidth-wide load
            // starting from start.  It's still generally worthwhile
            // issuing the load we've been considering, though, since it
            // will provide values for a number of previous offsets.  This
            // load will have one or more elements at the end of its range
            // that is not needed by any of the offsets under
            // consideration.  As such, there are three cases where issuing
            // this load is a bad idea:
            //
            // 1. 2-wide loads: we know that we haven't completely filled
            //    the 2-wide vector, since otherwise the if() test above
            //    would have succeeded previously.  Therefore, we must have
            //    a situation with offsets like (4,6,...); it would be a
            //    silly idea to issue a 2-wide load to get the value for
            //    the 4 offset, versus failing here and issuing a scalar
            //    load instead.
            //
            // 2. If there are too many unnecessary values at the end of
            //    the load extent (defined as more than half of them)--in
            //    this case, it'd be better to issue a vector load of
            //    smaller width anyway.
            //
            // 3. If the gap between the last accepted offset and the
            //    current one under consideration is more than the page
            //    size.  In this case we can't be sure whether or not some
            //    of the unused elements at the end of the load will
            //    straddle a page boundary and thus lead to an undesirable
            //    fault.  (It's hard to imagine this happening in practice,
            //    except under contrived circumstances, but better safe
            //    than sorry.)
            const int pageSize = 4096;
            if (vectorWidth != 2 && (lastAccepted - start) > (vectorWidth / 2) && (*iter - lastAccepted) < pageSize) {
                *newIter = iter;
                return true;
            } else
                return false;
        }

        // Continue moving forward
        lastAccepted = *iter;
        ++iter;
    }

    return false;
}

/** Given a set of offsets from a common base pointer that we need to get
    loaded into memory, determine a reasonable set of load operations that
    gets all of the corresponding values in memory (ideally, including as
    many as possible wider vector loads rather than scalar loads).  Return
    a CoalescedLoadOp for each one in the *loads array.
 */
static void lSelectLoads(const std::vector<int64_t> &loadOffsets, std::vector<CoalescedLoadOp> *loads) {
    // First, get a sorted set of unique offsets to load from.
    std::set<int64_t> allOffsets;
    for (unsigned int i = 0; i < loadOffsets.size(); ++i)
        allOffsets.insert(loadOffsets[i]);

    std::set<int64_t>::iterator iter = allOffsets.begin();
    while (iter != allOffsets.end()) {
        Debug(SourcePos(), "Load needed at %" PRId64 ".", *iter);
        ++iter;
    }

    // Now, iterate over the offsets from low to high.  Starting at the
    // current offset, we see if a vector load starting from that offset
    // will cover loads at subsequent offsets as well.
    iter = allOffsets.begin();
    while (iter != allOffsets.end()) {
        // Consider vector loads of width of each of the elements of
        // spanSizes[], in order.
        int vectorWidths[] = {8, 4, 2};
        int nVectorWidths = sizeof(vectorWidths) / sizeof(vectorWidths[0]);
        bool gotOne = false;
        for (int i = 0; i < nVectorWidths; ++i) {
            // See if a load of vector with width vectorWidths[i] would be
            // effective (i.e. would cover a reasonable number of the
            // offsets that need to be loaded from).
            std::set<int64_t>::iterator newIter;
            if (lVectorLoadIsEfficient(iter, allOffsets.end(), &newIter, vectorWidths[i])) {
                // Yes: create the corresponding coalesced load and update
                // the iterator to the returned iterator; doing so skips
                // over the additional offsets that are taken care of by
                // this load.
                loads->push_back(CoalescedLoadOp(*iter, vectorWidths[i]));
                iter = newIter;
                gotOne = true;
                break;
            }
        }

        if (gotOne == false) {
            // We couldn't find a vector load starting from this offset
            // that made sense, so emit a scalar load and continue onward.
            loads->push_back(CoalescedLoadOp(*iter, 1));
            ++iter;
        }
    }
}

/** Print a performance message with the details of the result of
    coalescing over a group of gathers. */
static void lCoalescePerfInfo(const std::vector<llvm::CallInst *> &coalesceGroup,
                              const std::vector<CoalescedLoadOp> &loadOps) {
    SourcePos pos;
    lGetSourcePosFromMetadata(coalesceGroup[0], &pos);

    // Create a string that indicates the line numbers of the subsequent
    // gathers from the first one that were coalesced here.
    char otherPositions[512];
    otherPositions[0] = '\0';
    if (coalesceGroup.size() > 1) {
        const char *plural = (coalesceGroup.size() > 2) ? "s" : "";
        char otherBuf[32];
        snprintf(otherBuf, sizeof(otherBuf), "(other%s at line%s ", plural, plural);
        strncat(otherPositions, otherBuf, sizeof(otherPositions) - strlen(otherPositions) - 1);

        for (int i = 1; i < (int)coalesceGroup.size(); ++i) {
            SourcePos p;
            bool ok = lGetSourcePosFromMetadata(coalesceGroup[i], &p);
            if (ok) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%d", p.first_line);
                strncat(otherPositions, buf, sizeof(otherPositions) - strlen(otherPositions) - 1);
                if (i < (int)coalesceGroup.size() - 1)
                    strncat(otherPositions, ", ", sizeof(otherPositions) - strlen(otherPositions) - 1);
            }
        }
        strncat(otherPositions, ") ", sizeof(otherPositions) - strlen(otherPositions) - 1);
    }

    // Count how many loads of each size there were.
    std::map<int, int> loadOpsCount;
    for (int i = 0; i < (int)loadOps.size(); ++i)
        ++loadOpsCount[loadOps[i].count];

    // Generate a string the describes the mix of load ops
    char loadOpsInfo[512];
    loadOpsInfo[0] = '\0';
    std::map<int, int>::const_iterator iter = loadOpsCount.begin();
    while (iter != loadOpsCount.end()) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d x %d-wide", iter->second, iter->first);
        if ((strlen(loadOpsInfo) + strlen(buf)) >= 512) {
            break;
        }
        strncat(loadOpsInfo, buf, sizeof(loadOpsInfo) - strlen(loadOpsInfo) - 1);
        ++iter;
        if (iter != loadOpsCount.end())
            strncat(loadOpsInfo, ", ", sizeof(loadOpsInfo) - strlen(loadOpsInfo) - 1);
    }

    if (g->opt.level > 0) {
        if (coalesceGroup.size() == 1)
            PerformanceWarning(pos, "Coalesced gather into %d load%s (%s).", (int)loadOps.size(),
                               (loadOps.size() > 1) ? "s" : "", loadOpsInfo);
        else
            PerformanceWarning(pos,
                               "Coalesced %d gathers starting here %sinto %d "
                               "load%s (%s).",
                               (int)coalesceGroup.size(), otherPositions, (int)loadOps.size(),
                               (loadOps.size() > 1) ? "s" : "", loadOpsInfo);
    }
}

/** Utility routine that computes an offset from a base pointer and then
    returns the result of a load of the given type from the resulting
    location:

    return *((type *)(basePtr + offset))
 */
llvm::Value *lGEPAndLoad(llvm::Value *basePtr, int64_t offset, int align, llvm::Instruction *insertBefore,
                         llvm::Type *type) {
    llvm::Value *ptr = lGEPInst(basePtr, LLVMInt64(offset), "new_base", insertBefore);
    ptr = new llvm::BitCastInst(ptr, llvm::PointerType::get(type, 0), "ptr_cast", insertBefore);
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
    return new llvm::LoadInst(ptr, "gather_load", false /* not volatile */, llvm::MaybeAlign(align), insertBefore);
#else // LLVM 11.0+
    Assert(llvm::isa<llvm::PointerType>(ptr->getType()));
    return new llvm::LoadInst(llvm::dyn_cast<llvm::PointerType>(ptr->getType())->getPointerElementType(), ptr,
                              "gather_load", false /* not volatile */, llvm::MaybeAlign(align).valueOrOne(),
                              insertBefore);
#endif
}

/* Having decided that we're doing to emit a series of loads, as encoded in
   the loadOps array, this function emits the corresponding load
   instructions.
 */
static void lEmitLoads(llvm::Value *basePtr, std::vector<CoalescedLoadOp> &loadOps, int elementSize,
                       llvm::Instruction *insertBefore) {
    Debug(SourcePos(), "Coalesce doing %d loads.", (int)loadOps.size());
    for (int i = 0; i < (int)loadOps.size(); ++i) {
        Debug(SourcePos(), "Load #%d @ %" PRId64 ", %d items", i, loadOps[i].start, loadOps[i].count);

        // basePtr is an i8 *, so the offset from it should be in terms of
        // bytes, not underlying i32 elements.
        int64_t start = loadOps[i].start * elementSize;

        int align = 4;
        switch (loadOps[i].count) {
        case 1:
            // Single 32-bit scalar load
            loadOps[i].load = lGEPAndLoad(basePtr, start, align, insertBefore, LLVMTypes::Int32Type);
            break;
        case 2: {
            // Emit 2 x i32 loads as i64 loads and then break the result
            // into two 32-bit parts.
            loadOps[i].load = lGEPAndLoad(basePtr, start, align, insertBefore, LLVMTypes::Int64Type);
            // element0 = (int32)value;
            loadOps[i].element0 =
                new llvm::TruncInst(loadOps[i].load, LLVMTypes::Int32Type, "load64_elt0", insertBefore);
            // element1 = (int32)(value >> 32)
            llvm::Value *shift = llvm::BinaryOperator::Create(llvm::Instruction::LShr, loadOps[i].load, LLVMInt64(32),
                                                              "load64_shift", insertBefore);
            loadOps[i].element1 = new llvm::TruncInst(shift, LLVMTypes::Int32Type, "load64_elt1", insertBefore);
            break;
        }
        case 4: {
            // 4-wide vector load
            if (g->opt.forceAlignedMemory) {
                align = g->target->getNativeVectorAlignment();
            }
            llvm::VectorType *vt = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
            loadOps[i].load = lGEPAndLoad(basePtr, start, align, insertBefore, vt);
            break;
        }
        case 8: {
            // 8-wide vector load
            if (g->opt.forceAlignedMemory) {
                align = g->target->getNativeVectorAlignment();
            }
            llvm::VectorType *vt = LLVMVECTOR::get(LLVMTypes::Int32Type, 8);
            loadOps[i].load = lGEPAndLoad(basePtr, start, align, insertBefore, vt);
            break;
        }
        default:
            FATAL("Unexpected load count in lEmitLoads()");
        }
    }
}

/** Convert any loads of 8-wide vectors into two 4-wide vectors
    (logically).  This allows the assembly code below to always operate on
    4-wide vectors, which leads to better code.  Returns a new vector of
    load operations.
 */
static std::vector<CoalescedLoadOp> lSplit8WideLoads(const std::vector<CoalescedLoadOp> &loadOps,
                                                     llvm::Instruction *insertBefore) {
    std::vector<CoalescedLoadOp> ret;
    for (unsigned int i = 0; i < loadOps.size(); ++i) {
        if (loadOps[i].count == 8) {
            // Create fake CoalescedLOadOps, where the load llvm::Value is
            // actually a shuffle that pulls either the first 4 or the last
            // 4 values out of the original 8-wide loaded value.
            int32_t shuf[2][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}};

            ret.push_back(CoalescedLoadOp(loadOps[i].start, 4));
            ret.back().load = LLVMShuffleVectors(loadOps[i].load, loadOps[i].load, shuf[0], 4, insertBefore);

            ret.push_back(CoalescedLoadOp(loadOps[i].start + 4, 4));
            ret.back().load = LLVMShuffleVectors(loadOps[i].load, loadOps[i].load, shuf[1], 4, insertBefore);
        } else
            ret.push_back(loadOps[i]);
    }

    return ret;
}

/** Given a 1-wide load of a 32-bit value, merge its value into the result
    vector for any and all elements for which it applies.
 */
static llvm::Value *lApplyLoad1(llvm::Value *result, const CoalescedLoadOp &load, const int64_t offsets[4], bool set[4],
                                llvm::Instruction *insertBefore) {
    for (int elt = 0; elt < 4; ++elt) {
        if (offsets[elt] >= load.start && offsets[elt] < load.start + load.count) {
            Debug(SourcePos(),
                  "Load 1 @ %" PRId64 " matches for element #%d "
                  "(value %" PRId64 ")",
                  load.start, elt, offsets[elt]);
            // If this load gives one of the values that we need, then we
            // can just insert it in directly
            Assert(set[elt] == false);
            result = llvm::InsertElementInst::Create(result, load.load, LLVMInt32(elt), "insert_load", insertBefore);
            set[elt] = true;
        }
    }

    return result;
}

/** Similarly, incorporate the values from a 2-wide load into any vector
    elements that they apply to. */
static llvm::Value *lApplyLoad2(llvm::Value *result, const CoalescedLoadOp &load, const int64_t offsets[4], bool set[4],
                                llvm::Instruction *insertBefore) {
    int elt = 0;
    while (elt < 4) {
        // First, try to do a 64-bit-wide insert into the result vector.
        // We can do this when we're currently at an even element, when the
        // current and next element have consecutive values, and where the
        // original 64-bit load is at the offset needed by the current
        // element.
        if ((elt & 1) == 0 && offsets[elt] + 1 == offsets[elt + 1] && offsets[elt] == load.start) {
            Debug(SourcePos(),
                  "Load 2 @ %" PRId64 " matches for elements #%d,%d "
                  "(values %" PRId64 ",%" PRId64 ")",
                  load.start, elt, elt + 1, offsets[elt], offsets[elt + 1]);
            Assert(set[elt] == false && ((elt < 3) && set[elt + 1] == false));

            // In this case, we bitcast from a 4xi32 to a 2xi64 vector
            llvm::Type *vec2x64Type = LLVMVECTOR::get(LLVMTypes::Int64Type, 2);
            result = new llvm::BitCastInst(result, vec2x64Type, "to2x64", insertBefore);

            // And now we can insert the 64-bit wide value into the
            // appropriate elment
            result = llvm::InsertElementInst::Create(result, load.load, LLVMInt32(elt / 2), "insert64", insertBefore);

            // And back to 4xi32.
            llvm::Type *vec4x32Type = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
            result = new llvm::BitCastInst(result, vec4x32Type, "to4x32", insertBefore);

            set[elt] = true;
            if (elt < 3) {
                set[elt + 1] = true;
            }
            // Advance elt one extra time, since we just took care of two
            // elements
            ++elt;
        } else if (offsets[elt] >= load.start && offsets[elt] < load.start + load.count) {
            Debug(SourcePos(),
                  "Load 2 @ %" PRId64 " matches for element #%d "
                  "(value %" PRId64 ")",
                  load.start, elt, offsets[elt]);
            // Otherwise, insert one of the 32-bit pieces into an element
            // of the final vector
            Assert(set[elt] == false);
            llvm::Value *toInsert = (offsets[elt] == load.start) ? load.element0 : load.element1;
            result = llvm::InsertElementInst::Create(result, toInsert, LLVMInt32(elt), "insert_load", insertBefore);
            set[elt] = true;
        }
        ++elt;
    }

    return result;
}

#if 1
/* This approach works better with AVX, while the #else path generates
   slightly better code with SSE.  Need to continue to dig into performance
   details with this stuff in general... */

/** And handle a 4-wide load */
static llvm::Value *lApplyLoad4(llvm::Value *result, const CoalescedLoadOp &load, const int64_t offsets[4], bool set[4],
                                llvm::Instruction *insertBefore) {
    // Conceptually, we're doing to consider doing a shuffle vector with
    // the 4-wide load and the 4-wide result we have so far to generate a
    // new 4-wide vector.  We'll start with shuffle indices that just
    // select each element of the result so far for the result.
    int32_t shuf[4] = {4, 5, 6, 7};

    for (int elt = 0; elt < 4; ++elt) {
        if (offsets[elt] >= load.start && offsets[elt] < load.start + load.count) {
            Debug(SourcePos(),
                  "Load 4 @ %" PRId64 " matches for element #%d "
                  "(value %" PRId64 ")",
                  load.start, elt, offsets[elt]);

            // If the current element falls within the range of locations
            // that the 4-wide load covers, then compute the appropriate
            // shuffle index that extracts the appropriate element from the
            // load.
            Assert(set[elt] == false);
            shuf[elt] = int32_t(offsets[elt] - load.start);
            set[elt] = true;
        }
    }

    // Now, issue a shufflevector instruction if any of the values from the
    // load we just considered were applicable.
    if (shuf[0] != 4 || shuf[1] != 5 || shuf[2] != 6 || shuf[3] != 7)
        result = LLVMShuffleVectors(load.load, result, shuf, 4, insertBefore);

    return result;
}

/** We're need to fill in the values for a 4-wide result vector.  This
    function looks at all of the generated loads and extracts the
    appropriate elements from the appropriate loads to assemble the result.
    Here the offsets[] parameter gives the 4 offsets from the base pointer
    for the four elements of the result.
*/
static llvm::Value *lAssemble4Vector(const std::vector<CoalescedLoadOp> &loadOps, const int64_t offsets[4],
                                     llvm::Instruction *insertBefore) {
    llvm::Type *returnType = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
    llvm::Value *result = llvm::UndefValue::get(returnType);

    Debug(SourcePos(), "Starting search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    // Track whether we have found a valid value for each of the four
    // elements of the result
    bool set[4] = {false, false, false, false};

    // Loop over all of the loads and check each one to see if it provides
    // a value that's applicable to the result
    for (int load = 0; load < (int)loadOps.size(); ++load) {
        const CoalescedLoadOp &li = loadOps[load];

        switch (li.count) {
        case 1:
            result = lApplyLoad1(result, li, offsets, set, insertBefore);
            break;
        case 2:
            result = lApplyLoad2(result, li, offsets, set, insertBefore);
            break;
        case 4:
            result = lApplyLoad4(result, li, offsets, set, insertBefore);
            break;
        default:
            FATAL("Unexpected load count in lAssemble4Vector()");
        }
    }

    Debug(SourcePos(), "Done with search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    for (int i = 0; i < 4; ++i)
        Assert(set[i] == true);

    return result;
}

#else

static llvm::Value *lApplyLoad4s(llvm::Value *result, const std::vector<CoalescedLoadOp> &loadOps,
                                 const int64_t offsets[4], bool set[4], llvm::Instruction *insertBefore) {
    int32_t firstMatchElements[4] = {-1, -1, -1, -1};
    const CoalescedLoadOp *firstMatch = NULL;

    Assert(llvm::isa<llvm::UndefValue>(result));

    for (int load = 0; load < (int)loadOps.size(); ++load) {
        const CoalescedLoadOp &loadop = loadOps[load];
        if (loadop.count != 4)
            continue;

        int32_t matchElements[4] = {-1, -1, -1, -1};
        bool anyMatched = false;
        for (int elt = 0; elt < 4; ++elt) {
            if (offsets[elt] >= loadop.start && offsets[elt] < loadop.start + loadop.count) {
                Debug(SourcePos(),
                      "Load 4 @ %" PRId64 " matches for element #%d "
                      "(value %" PRId64 ")",
                      loadop.start, elt, offsets[elt]);
                anyMatched = true;
                Assert(set[elt] == false);
                matchElements[elt] = offsets[elt] - loadop.start;
                set[elt] = true;
            }
        }

        if (anyMatched) {
            if (llvm::isa<llvm::UndefValue>(result)) {
                if (firstMatch == NULL) {
                    firstMatch = &loadop;
                    for (int i = 0; i < 4; ++i)
                        firstMatchElements[i] = matchElements[i];
                } else {
                    int32_t shuffle[4] = {-1, -1, -1, -1};
                    for (int i = 0; i < 4; ++i) {
                        if (firstMatchElements[i] != -1)
                            shuffle[i] = firstMatchElements[i];
                        else
                            shuffle[i] = 4 + matchElements[i];
                    }
                    result = LLVMShuffleVectors(firstMatch->load, loadop.load, shuffle, 4, insertBefore);
                    firstMatch = NULL;
                }
            } else {
                int32_t shuffle[4] = {-1, -1, -1, -1};
                for (int i = 0; i < 4; ++i) {
                    if (matchElements[i] != -1)
                        shuffle[i] = 4 + matchElements[i];
                    else
                        shuffle[i] = i;
                }
                result = LLVMShuffleVectors(result, loadop.load, shuffle, 4, insertBefore);
            }
        }
    }

    if (firstMatch != NULL && llvm::isa<llvm::UndefValue>(result))
        return LLVMShuffleVectors(firstMatch->load, result, firstMatchElements, 4, insertBefore);
    else
        return result;
}

static llvm::Value *lApplyLoad12s(llvm::Value *result, const std::vector<CoalescedLoadOp> &loadOps,
                                  const int64_t offsets[4], bool set[4], llvm::Instruction *insertBefore) {
    // Loop over all of the loads and check each one to see if it provides
    // a value that's applicable to the result
    for (int load = 0; load < (int)loadOps.size(); ++load) {
        const CoalescedLoadOp &loadop = loadOps[load];
        Assert(loadop.count == 1 || loadop.count == 2 || loadop.count == 4);

        if (loadop.count == 1)
            result = lApplyLoad1(result, loadop, offsets, set, insertBefore);
        else if (loadop.count == 2)
            result = lApplyLoad2(result, loadop, offsets, set, insertBefore);
    }
    return result;
}

/** We're need to fill in the values for a 4-wide result vector.  This
    function looks at all of the generated loads and extracts the
    appropriate elements from the appropriate loads to assemble the result.
    Here the offsets[] parameter gives the 4 offsets from the base pointer
    for the four elements of the result.
*/
static llvm::Value *lAssemble4Vector(const std::vector<CoalescedLoadOp> &loadOps, const int64_t offsets[4],
                                     llvm::Instruction *insertBefore) {
    llvm::Type *returnType = LLVMVECTOR::get(LLVMTypes::Int32Type, 4);
    llvm::Value *result = llvm::UndefValue::get(returnType);

    Debug(SourcePos(), "Starting search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    // Track whether we have found a valid value for each of the four
    // elements of the result
    bool set[4] = {false, false, false, false};

    result = lApplyLoad4s(result, loadOps, offsets, set, insertBefore);
    result = lApplyLoad12s(result, loadOps, offsets, set, insertBefore);

    Debug(SourcePos(), "Done with search for loads [%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "].", offsets[0],
          offsets[1], offsets[2], offsets[3]);

    for (int i = 0; i < 4; ++i)
        Assert(set[i] == true);

    return result;
}
#endif

/** Given the set of loads that we've done and the set of result values to
    be computed, this function computes the final llvm::Value *s for each
    result vector.
 */
static void lAssembleResultVectors(const std::vector<CoalescedLoadOp> &loadOps,
                                   const std::vector<int64_t> &constOffsets, std::vector<llvm::Value *> &results,
                                   llvm::Instruction *insertBefore) {
    // We work on 4-wide chunks of the final values, even when we're
    // computing 8-wide or 16-wide vectors.  This gives better code from
    // LLVM's SSE/AVX code generators.
    Assert((constOffsets.size() % 4) == 0);
    std::vector<llvm::Value *> vec4s;
    for (int i = 0; i < (int)constOffsets.size(); i += 4)
        vec4s.push_back(lAssemble4Vector(loadOps, &constOffsets[i], insertBefore));

    // And now concatenate 1, 2, or 4 of the 4-wide vectors computed above
    // into 4, 8, or 16-wide final result vectors.
    int numGathers = constOffsets.size() / g->target->getVectorWidth();
    for (int i = 0; i < numGathers; ++i) {
        llvm::Value *result = NULL;
        switch (g->target->getVectorWidth()) {
        case 4:
            result = vec4s[i];
            break;
        case 8:
            result = LLVMConcatVectors(vec4s[2 * i], vec4s[2 * i + 1], insertBefore);
            break;
        case 16: {
            llvm::Value *v1 = LLVMConcatVectors(vec4s[4 * i], vec4s[4 * i + 1], insertBefore);
            llvm::Value *v2 = LLVMConcatVectors(vec4s[4 * i + 2], vec4s[4 * i + 3], insertBefore);
            result = LLVMConcatVectors(v1, v2, insertBefore);
            break;
        }
        default:
            FATAL("Unhandled vector width in lAssembleResultVectors()");
        }

        results.push_back(result);
    }
}

/** Given a call to a gather function, extract the base pointer, the 2/4/8
    scale, and the first varying offsets value to use them to compute that
    scalar base pointer that is shared by all of the gathers in the group.
    (Thus, this base pointer plus the constant offsets term for each gather
    gives the set of addresses to use for each gather.
 */
static llvm::Value *lComputeBasePtr(llvm::CallInst *gatherInst, llvm::Instruction *insertBefore) {
    llvm::Value *basePtr = gatherInst->getArgOperand(0);
    llvm::Value *variableOffsets = gatherInst->getArgOperand(1);
    llvm::Value *offsetScale = gatherInst->getArgOperand(2);

    // All of the variable offsets values should be the same, due to
    // checking for this in GatherCoalescePass::runOnBasicBlock().  Thus,
    // extract the first value and use that as a scalar.
    llvm::Value *variable = LLVMExtractFirstVectorElement(variableOffsets);
    Assert(variable != NULL);
    if (variable->getType() == LLVMTypes::Int64Type)
        offsetScale = new llvm::ZExtInst(offsetScale, LLVMTypes::Int64Type, "scale_to64", insertBefore);
    llvm::Value *offset =
        llvm::BinaryOperator::Create(llvm::Instruction::Mul, variable, offsetScale, "offset", insertBefore);

    return lGEPInst(basePtr, offset, "new_base", insertBefore);
}

/** Extract the constant offsets (from the common base pointer) from each
    of the gathers in a set to be coalesced.  These come in as byte
    offsets, but we'll transform them into offsets in terms of the size of
    the base scalar type being gathered.  (e.g. for an i32 gather, we might
    have offsets like <0,4,16,20>, which would be transformed to <0,1,4,5>
    here.)
 */
static void lExtractConstOffsets(const std::vector<llvm::CallInst *> &coalesceGroup, int elementSize,
                                 std::vector<int64_t> *constOffsets) {
    int width = g->target->getVectorWidth();
    *constOffsets = std::vector<int64_t>(coalesceGroup.size() * width, 0);

    int64_t *endPtr = &((*constOffsets)[0]);
    for (int i = 0; i < (int)coalesceGroup.size(); ++i, endPtr += width) {
        llvm::Value *offsets = coalesceGroup[i]->getArgOperand(3);
        int nElts;
        bool ok = LLVMExtractVectorInts(offsets, endPtr, &nElts);
        Assert(ok && nElts == width);
    }

    for (int i = 0; i < (int)constOffsets->size(); ++i)
        (*constOffsets)[i] /= elementSize;
}

/** Actually do the coalescing.  We have a set of gathers all accessing
    addresses of the form:

    (ptr + {1,2,4,8} * varyingOffset) + constOffset, a.k.a.
    basePtr + constOffset

    where varyingOffset actually has the same value across all of the SIMD
    lanes and where the part in parenthesis has the same value for all of
    the gathers in the group.
 */
static bool lCoalesceGathers(const std::vector<llvm::CallInst *> &coalesceGroup) {
    llvm::Instruction *insertBefore = coalesceGroup[0];

    // First, compute the shared base pointer for all of the gathers
    llvm::Value *basePtr = lComputeBasePtr(coalesceGroup[0], insertBefore);

    int elementSize = 0;
    if (coalesceGroup[0]->getType() == LLVMTypes::Int32VectorType ||
        coalesceGroup[0]->getType() == LLVMTypes::FloatVectorType)
        elementSize = 4;
    else if (coalesceGroup[0]->getType() == LLVMTypes::Int64VectorType ||
             coalesceGroup[0]->getType() == LLVMTypes::DoubleVectorType)
        elementSize = 8;
    else
        FATAL("Unexpected gather type in lCoalesceGathers");

    // Extract the constant offsets from the gathers into the constOffsets
    // vector: the first vectorWidth elements will be those for the first
    // gather, the next vectorWidth those for the next gather, and so
    // forth.
    std::vector<int64_t> constOffsets;
    lExtractConstOffsets(coalesceGroup, elementSize, &constOffsets);

    // Determine a set of loads to perform to get all of the values we need
    // loaded.
    std::vector<CoalescedLoadOp> loadOps;
    lSelectLoads(constOffsets, &loadOps);

    lCoalescePerfInfo(coalesceGroup, loadOps);

    // Actually emit load instructions for them
    lEmitLoads(basePtr, loadOps, elementSize, insertBefore);

    // Now, for any loads that give us <8 x i32> vectors, split their
    // values into two <4 x i32> vectors; it turns out that LLVM gives us
    // better code on AVX when we assemble the pieces from 4-wide vectors.
    loadOps = lSplit8WideLoads(loadOps, insertBefore);

    // Given all of these chunks of values, shuffle together a vector that
    // gives us each result value; the i'th element of results[] gives the
    // result for the i'th gather in coalesceGroup.
    std::vector<llvm::Value *> results;
    lAssembleResultVectors(loadOps, constOffsets, results, insertBefore);

    // Finally, replace each of the original gathers with the instruction
    // that gives the value from the coalescing process.
    Assert(results.size() == coalesceGroup.size());
    for (int i = 0; i < (int)results.size(); ++i) {
        llvm::Instruction *ir = llvm::dyn_cast<llvm::Instruction>(results[i]);
        Assert(ir != NULL);

        llvm::Type *origType = coalesceGroup[i]->getType();
        if (origType != ir->getType())
            ir = new llvm::BitCastInst(ir, origType, ir->getName(), coalesceGroup[i]);

        // Previously, all of the instructions to compute the final result
        // were into the basic block here; here we remove the very last one
        // of them (that holds the final result) from the basic block.
        // This way, the following ReplaceInstWithInst() call will operate
        // successfully. (It expects that the second argument not be in any
        // basic block.)
        ir->removeFromParent();

        llvm::ReplaceInstWithInst(coalesceGroup[i], ir);
    }

    return true;
}

/** Given an instruction, returns true if the instructon may write to
    memory.  This is a conservative test in that it may return true for
    some instructions that don't actually end up writing to memory, but
    should never return false for an instruction that does write to
    memory. */
static bool lInstructionMayWriteToMemory(llvm::Instruction *inst) {
    if (llvm::isa<llvm::StoreInst>(inst) || llvm::isa<llvm::AtomicRMWInst>(inst) ||
        llvm::isa<llvm::AtomicCmpXchgInst>(inst))
        // FIXME: we could be less conservative and try to allow stores if
        // we are sure that the pointers don't overlap..
        return true;

    // Otherwise, any call instruction that doesn't have an attribute
    // indicating it won't write to memory has to be treated as a potential
    // store.
    llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst);
    if (ci != NULL) {
        llvm::Function *calledFunc = ci->getCalledFunction();
        if (calledFunc == NULL)
            return true;

        if (calledFunc->onlyReadsMemory() || calledFunc->doesNotAccessMemory())
            return false;
        return true;
    }

    return false;
}

bool GatherCoalescePass::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("GatherCoalescePass");

    llvm::Function *gatherFuncs[] = {
        m->module->getFunction("__pseudo_gather_factored_base_offsets32_i32"),
        m->module->getFunction("__pseudo_gather_factored_base_offsets32_float"),
        m->module->getFunction("__pseudo_gather_factored_base_offsets64_i32"),
        m->module->getFunction("__pseudo_gather_factored_base_offsets64_float"),
    };
    int nGatherFuncs = sizeof(gatherFuncs) / sizeof(gatherFuncs[0]);

    bool modifiedAny = false;

restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        // Iterate over all of the instructions and look for calls to
        // __pseudo_gather_factored_base_offsets{32,64}_{i32,float} calls.
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst == NULL)
            continue;

        llvm::Function *calledFunc = callInst->getCalledFunction();
        if (calledFunc == NULL)
            continue;

        int i;
        for (i = 0; i < nGatherFuncs; ++i)
            if (gatherFuncs[i] != NULL && calledFunc == gatherFuncs[i])
                break;
        if (i == nGatherFuncs)
            // Doesn't match any of the types of gathers we care about
            continue;

        SourcePos pos;
        lGetSourcePosFromMetadata(callInst, &pos);
        Debug(pos, "Checking for coalescable gathers starting here...");

        llvm::Value *base = callInst->getArgOperand(0);
        llvm::Value *variableOffsets = callInst->getArgOperand(1);
        llvm::Value *offsetScale = callInst->getArgOperand(2);
        llvm::Value *mask = callInst->getArgOperand(4);

        // To apply this optimization, we need a set of one or more gathers
        // that fulfill the following conditions:
        //
        // - Mask all on
        // - The variable offsets to all have the same value (i.e., to be
        //   uniform).
        // - Same base pointer, variable offsets, and offset scale (for
        //   more than one gather)
        //
        // Then and only then do we have a common base pointer with all
        // offsets from that constants (in which case we can potentially
        // coalesce).
        if (lGetMaskStatus(mask) != MaskStatus::all_on)
            continue;

        if (!LLVMVectorValuesAllEqual(variableOffsets))
            continue;

        // coalesceGroup stores the set of gathers that we're going to try to
        // coalesce over
        std::vector<llvm::CallInst *> coalesceGroup;
        coalesceGroup.push_back(callInst);

        // Start iterating at the instruction after the initial gather;
        // look at the remainder of instructions in the basic block (up
        // until we reach a write to memory) to try to find any other
        // gathers that can coalesce with this one.
        llvm::BasicBlock::iterator fwdIter = iter;
        ++fwdIter;
        for (; fwdIter != bb.end(); ++fwdIter) {
            // Must stop once we come to an instruction that may write to
            // memory; otherwise we could end up moving a read before this
            // write.
            if (lInstructionMayWriteToMemory(&*fwdIter))
                break;

            llvm::CallInst *fwdCall = llvm::dyn_cast<llvm::CallInst>(&*fwdIter);
            if (fwdCall == NULL || fwdCall->getCalledFunction() != calledFunc)
                continue;

            SourcePos fwdPos;
            // TODO: need to redesign metadata attached to pseudo calls,
            // LLVM drops metadata frequently and it results in bad disgnostics.
            lGetSourcePosFromMetadata(fwdCall, &fwdPos);

#ifndef ISPC_NO_DUMPS
            if (g->debugPrint) {
                if (base != fwdCall->getArgOperand(0)) {
                    Debug(fwdPos, "base pointers mismatch");
                    LLVMDumpValue(base);
                    LLVMDumpValue(fwdCall->getArgOperand(0));
                }
                if (variableOffsets != fwdCall->getArgOperand(1)) {
                    Debug(fwdPos, "varying offsets mismatch");
                    LLVMDumpValue(variableOffsets);
                    LLVMDumpValue(fwdCall->getArgOperand(1));
                }
                if (offsetScale != fwdCall->getArgOperand(2)) {
                    Debug(fwdPos, "offset scales mismatch");
                    LLVMDumpValue(offsetScale);
                    LLVMDumpValue(fwdCall->getArgOperand(2));
                }
                if (mask != fwdCall->getArgOperand(4)) {
                    Debug(fwdPos, "masks mismatch");
                    LLVMDumpValue(mask);
                    LLVMDumpValue(fwdCall->getArgOperand(4));
                }
            }
#endif

            if (base == fwdCall->getArgOperand(0) && variableOffsets == fwdCall->getArgOperand(1) &&
                offsetScale == fwdCall->getArgOperand(2) && mask == fwdCall->getArgOperand(4)) {
                Debug(fwdPos, "This gather can be coalesced.");
                coalesceGroup.push_back(fwdCall);

                if (coalesceGroup.size() == 4)
                    // FIXME: untested heuristic: don't try to coalesce
                    // over a window of more than 4 gathers, so that we
                    // don't cause too much register pressure and end up
                    // spilling to memory anyway.
                    break;
            } else
                Debug(fwdPos, "This gather doesn't match the initial one.");
        }

        Debug(pos, "Done with checking for matching gathers");

        // Now that we have a group of gathers, see if we can coalesce them
        // into something more efficient than the original set of gathers.
        if (lCoalesceGathers(coalesceGroup)) {
            modifiedAny = true;
            goto restart;
        }
    }

    DEBUG_END_PASS("GatherCoalescePass");

    return modifiedAny;
}

bool GatherCoalescePass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("GatherCoalescePass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateGatherCoalescePass() { return new GatherCoalescePass; }

///////////////////////////////////////////////////////////////////////////
// ReplacePseudoMemoryOpsPass

/** For any gathers and scatters remaining after the GSToLoadStorePass
    runs, we need to turn them into actual native gathers and scatters.
    This task is handled by the ReplacePseudoMemoryOpsPass here.
 */
class ReplacePseudoMemoryOpsPass : public llvm::FunctionPass {
  public:
    static char ID;
    ReplacePseudoMemoryOpsPass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Replace Pseudo Memory Ops"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);
};

char ReplacePseudoMemoryOpsPass::ID = 0;

/** This routine attempts to determine if the given pointer in lvalue is
    pointing to stack-allocated memory.  It's conservative in that it
    should never return true for non-stack allocated memory, but may return
    false for memory that actually is stack allocated.  The basic strategy
    is to traverse through the operands and see if the pointer originally
    comes from an AllocaInst.
*/
static bool lIsSafeToBlend(llvm::Value *lvalue) {
    llvm::BitCastInst *bc = llvm::dyn_cast<llvm::BitCastInst>(lvalue);
    if (bc != NULL)
        return lIsSafeToBlend(bc->getOperand(0));
    else {
        llvm::AllocaInst *ai = llvm::dyn_cast<llvm::AllocaInst>(lvalue);
        if (ai) {
            llvm::Type *type = ai->getType();
            llvm::PointerType *pt = llvm::dyn_cast<llvm::PointerType>(type);
            Assert(pt != NULL);
            type = pt->PTR_ELT_TYPE();
            llvm::ArrayType *at;
            while ((at = llvm::dyn_cast<llvm::ArrayType>(type))) {
                type = at->getElementType();
            }
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
            llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(type);
#else
            llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(type);
#endif
            return (vt != NULL && (int)vt->getNumElements() == g->target->getVectorWidth());
        } else {
            llvm::GetElementPtrInst *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(lvalue);
            if (gep != NULL)
                return lIsSafeToBlend(gep->getOperand(0));
            else
                return false;
        }
    }
}

static bool lReplacePseudoMaskedStore(llvm::CallInst *callInst) {
    struct LMSInfo {
        LMSInfo(const char *pname, const char *bname, const char *msname) {
            pseudoFunc = m->module->getFunction(pname);
            blendFunc = m->module->getFunction(bname);
            maskedStoreFunc = m->module->getFunction(msname);
            Assert(pseudoFunc != NULL && blendFunc != NULL && maskedStoreFunc != NULL);
        }
        llvm::Function *pseudoFunc;
        llvm::Function *blendFunc;
        llvm::Function *maskedStoreFunc;
    };

    LMSInfo msInfo[] = {
        LMSInfo("__pseudo_masked_store_i8", "__masked_store_blend_i8", "__masked_store_i8"),
        LMSInfo("__pseudo_masked_store_i16", "__masked_store_blend_i16", "__masked_store_i16"),
        LMSInfo("__pseudo_masked_store_half", "__masked_store_blend_half", "__masked_store_half"),
        LMSInfo("__pseudo_masked_store_i32", "__masked_store_blend_i32", "__masked_store_i32"),
        LMSInfo("__pseudo_masked_store_float", "__masked_store_blend_float", "__masked_store_float"),
        LMSInfo("__pseudo_masked_store_i64", "__masked_store_blend_i64", "__masked_store_i64"),
        LMSInfo("__pseudo_masked_store_double", "__masked_store_blend_double", "__masked_store_double")};
    LMSInfo *info = NULL;
    for (unsigned int i = 0; i < sizeof(msInfo) / sizeof(msInfo[0]); ++i) {
        if (msInfo[i].pseudoFunc != NULL && callInst->getCalledFunction() == msInfo[i].pseudoFunc) {
            info = &msInfo[i];
            break;
        }
    }
    if (info == NULL)
        return false;

    llvm::Value *lvalue = callInst->getArgOperand(0);
    llvm::Value *rvalue = callInst->getArgOperand(1);
    llvm::Value *mask = callInst->getArgOperand(2);

    // We need to choose between doing the load + blend + store trick,
    // or serializing the masked store.  Even on targets with a native
    // masked store instruction, this is preferable since it lets us
    // keep values in registers rather than going out to the stack.
    bool doBlend = (!g->opt.disableBlendedMaskedStores && lIsSafeToBlend(lvalue));

    // Generate the call to the appropriate masked store function and
    // replace the __pseudo_* one with it.
    llvm::Function *fms = doBlend ? info->blendFunc : info->maskedStoreFunc;
    llvm::Instruction *inst = lCallInst(fms, lvalue, rvalue, mask, "", callInst);
    lCopyMetadata(inst, callInst);

    callInst->eraseFromParent();
    return true;
}

static bool lReplacePseudoGS(llvm::CallInst *callInst) {
    struct LowerGSInfo {
        LowerGSInfo(const char *pName, const char *aName, bool ig, bool ip) : isGather(ig), isPrefetch(ip) {
            pseudoFunc = m->module->getFunction(pName);
            actualFunc = m->module->getFunction(aName);
        }
        llvm::Function *pseudoFunc;
        llvm::Function *actualFunc;
        const bool isGather;
        const bool isPrefetch;
    };

    LowerGSInfo lgsInfo[] = {
        LowerGSInfo("__pseudo_gather32_i8", "__gather32_i8", true, false),
        LowerGSInfo("__pseudo_gather32_i16", "__gather32_i16", true, false),
        LowerGSInfo("__pseudo_gather32_half", "__gather32_half", true, false),
        LowerGSInfo("__pseudo_gather32_i32", "__gather32_i32", true, false),
        LowerGSInfo("__pseudo_gather32_float", "__gather32_float", true, false),
        LowerGSInfo("__pseudo_gather32_i64", "__gather32_i64", true, false),
        LowerGSInfo("__pseudo_gather32_double", "__gather32_double", true, false),

        LowerGSInfo("__pseudo_gather64_i8", "__gather64_i8", true, false),
        LowerGSInfo("__pseudo_gather64_i16", "__gather64_i16", true, false),
        LowerGSInfo("__pseudo_gather64_half", "__gather64_half", true, false),
        LowerGSInfo("__pseudo_gather64_i32", "__gather64_i32", true, false),
        LowerGSInfo("__pseudo_gather64_float", "__gather64_float", true, false),
        LowerGSInfo("__pseudo_gather64_i64", "__gather64_i64", true, false),
        LowerGSInfo("__pseudo_gather64_double", "__gather64_double", true, false),

        LowerGSInfo("__pseudo_gather_factored_base_offsets32_i8", "__gather_factored_base_offsets32_i8", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_i16", "__gather_factored_base_offsets32_i16", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_half", "__gather_factored_base_offsets32_half", true,
                    false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_i32", "__gather_factored_base_offsets32_i32", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_float", "__gather_factored_base_offsets32_float", true,
                    false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_i64", "__gather_factored_base_offsets32_i64", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_double", "__gather_factored_base_offsets32_double", true,
                    false),

        LowerGSInfo("__pseudo_gather_factored_base_offsets64_i8", "__gather_factored_base_offsets64_i8", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_i16", "__gather_factored_base_offsets64_i16", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_half", "__gather_factored_base_offsets64_half", true,
                    false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_i32", "__gather_factored_base_offsets64_i32", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_float", "__gather_factored_base_offsets64_float", true,
                    false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_i64", "__gather_factored_base_offsets64_i64", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_double", "__gather_factored_base_offsets64_double", true,
                    false),

        LowerGSInfo("__pseudo_gather_base_offsets32_i8", "__gather_base_offsets32_i8", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_i16", "__gather_base_offsets32_i16", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_half", "__gather_base_offsets32_half", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_i32", "__gather_base_offsets32_i32", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_float", "__gather_base_offsets32_float", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_i64", "__gather_base_offsets32_i64", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_double", "__gather_base_offsets32_double", true, false),

        LowerGSInfo("__pseudo_gather_base_offsets64_i8", "__gather_base_offsets64_i8", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_i16", "__gather_base_offsets64_i16", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_half", "__gather_base_offsets64_half", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_i32", "__gather_base_offsets64_i32", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_float", "__gather_base_offsets64_float", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_i64", "__gather_base_offsets64_i64", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_double", "__gather_base_offsets64_double", true, false),

        LowerGSInfo("__pseudo_scatter32_i8", "__scatter32_i8", false, false),
        LowerGSInfo("__pseudo_scatter32_i16", "__scatter32_i16", false, false),
        LowerGSInfo("__pseudo_scatter32_half", "__scatter32_half", false, false),
        LowerGSInfo("__pseudo_scatter32_i32", "__scatter32_i32", false, false),
        LowerGSInfo("__pseudo_scatter32_float", "__scatter32_float", false, false),
        LowerGSInfo("__pseudo_scatter32_i64", "__scatter32_i64", false, false),
        LowerGSInfo("__pseudo_scatter32_double", "__scatter32_double", false, false),

        LowerGSInfo("__pseudo_scatter64_i8", "__scatter64_i8", false, false),
        LowerGSInfo("__pseudo_scatter64_i16", "__scatter64_i16", false, false),
        LowerGSInfo("__pseudo_scatter64_half", "__scatter64_half", false, false),
        LowerGSInfo("__pseudo_scatter64_i32", "__scatter64_i32", false, false),
        LowerGSInfo("__pseudo_scatter64_float", "__scatter64_float", false, false),
        LowerGSInfo("__pseudo_scatter64_i64", "__scatter64_i64", false, false),
        LowerGSInfo("__pseudo_scatter64_double", "__scatter64_double", false, false),

        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_i8", "__scatter_factored_base_offsets32_i8", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_i16", "__scatter_factored_base_offsets32_i16", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_half", "__scatter_factored_base_offsets32_half", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_i32", "__scatter_factored_base_offsets32_i32", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_float", "__scatter_factored_base_offsets32_float", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_i64", "__scatter_factored_base_offsets32_i64", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_double", "__scatter_factored_base_offsets32_double",
                    false, false),

        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_i8", "__scatter_factored_base_offsets64_i8", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_i16", "__scatter_factored_base_offsets64_i16", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_half", "__scatter_factored_base_offsets64_half", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_i32", "__scatter_factored_base_offsets64_i32", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_float", "__scatter_factored_base_offsets64_float", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_i64", "__scatter_factored_base_offsets64_i64", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_double", "__scatter_factored_base_offsets64_double",
                    false, false),

        LowerGSInfo("__pseudo_scatter_base_offsets32_i8", "__scatter_base_offsets32_i8", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_i16", "__scatter_base_offsets32_i16", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_half", "__scatter_base_offsets32_half", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_i32", "__scatter_base_offsets32_i32", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_float", "__scatter_base_offsets32_float", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_i64", "__scatter_base_offsets32_i64", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_double", "__scatter_base_offsets32_double", false, false),

        LowerGSInfo("__pseudo_scatter_base_offsets64_i8", "__scatter_base_offsets64_i8", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_i16", "__scatter_base_offsets64_i16", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_half", "__scatter_base_offsets64_half", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_i32", "__scatter_base_offsets64_i32", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_float", "__scatter_base_offsets64_float", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_i64", "__scatter_base_offsets64_i64", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_double", "__scatter_base_offsets64_double", false, false),

        LowerGSInfo("__pseudo_prefetch_read_varying_1", "__prefetch_read_varying_1", false, true),
        LowerGSInfo("__pseudo_prefetch_read_varying_1_native", "__prefetch_read_varying_1_native", false, true),

        LowerGSInfo("__pseudo_prefetch_read_varying_2", "__prefetch_read_varying_2", false, true),
        LowerGSInfo("__pseudo_prefetch_read_varying_2_native", "__prefetch_read_varying_2_native", false, true),

        LowerGSInfo("__pseudo_prefetch_read_varying_3", "__prefetch_read_varying_3", false, true),
        LowerGSInfo("__pseudo_prefetch_read_varying_3_native", "__prefetch_read_varying_3_native", false, true),

        LowerGSInfo("__pseudo_prefetch_read_varying_nt", "__prefetch_read_varying_nt", false, true),
        LowerGSInfo("__pseudo_prefetch_read_varying_nt_native", "__prefetch_read_varying_nt_native", false, true),

        LowerGSInfo("__pseudo_prefetch_write_varying_1", "__prefetch_write_varying_1", false, true),
        LowerGSInfo("__pseudo_prefetch_write_varying_1_native", "__prefetch_write_varying_1_native", false, true),

        LowerGSInfo("__pseudo_prefetch_write_varying_2", "__prefetch_write_varying_2", false, true),
        LowerGSInfo("__pseudo_prefetch_write_varying_2_native", "__prefetch_write_varying_2_native", false, true),

        LowerGSInfo("__pseudo_prefetch_write_varying_3", "__prefetch_write_varying_3", false, true),
        LowerGSInfo("__pseudo_prefetch_write_varying_3_native", "__prefetch_write_varying_3_native", false, true),
    };

    llvm::Function *calledFunc = callInst->getCalledFunction();

    LowerGSInfo *info = NULL;
    for (unsigned int i = 0; i < sizeof(lgsInfo) / sizeof(lgsInfo[0]); ++i) {
        if (lgsInfo[i].pseudoFunc != NULL && calledFunc == lgsInfo[i].pseudoFunc) {
            info = &lgsInfo[i];
            break;
        }
    }
    if (info == NULL)
        return false;

    Assert(info->actualFunc != NULL);

    // Get the source position from the metadata attached to the call
    // instruction so that we can issue PerformanceWarning()s below.
    SourcePos pos;
    bool gotPosition = lGetSourcePosFromMetadata(callInst, &pos);

    callInst->setCalledFunction(info->actualFunc);
    // Check for alloca and if not alloca - generate __gather and change arguments
    if (gotPosition && (g->target->getVectorWidth() > 1) && (g->opt.level > 0)) {
        if (info->isGather)
            PerformanceWarning(pos, "Gather required to load value.");
        else if (!info->isPrefetch)
            PerformanceWarning(pos, "Scatter required to store value.");
    }
    return true;
}

bool ReplacePseudoMemoryOpsPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("ReplacePseudoMemoryOpsPass");

    bool modifiedAny = false;

restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst == NULL || callInst->getCalledFunction() == NULL)
            continue;

        if (lReplacePseudoGS(callInst)) {
            modifiedAny = true;
            goto restart;
        } else if (lReplacePseudoMaskedStore(callInst)) {
            modifiedAny = true;
            goto restart;
        }
    }

    DEBUG_END_PASS("ReplacePseudoMemoryOpsPass");

    return modifiedAny;
}

bool ReplacePseudoMemoryOpsPass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("ReplacePseudoMemoryOpsPass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateReplacePseudoMemoryOpsPass() { return new ReplacePseudoMemoryOpsPass; }

///////////////////////////////////////////////////////////////////////////
// IsCompileTimeConstantPass

/** LLVM IR implementations of target-specific functions may include calls
    to the functions "bool __is_compile_time_constant_*(...)"; these allow
    them to have specialied code paths for where the corresponding value is
    known at compile time.  For masks, for example, this allows them to not
    incur the cost of a MOVMSK call at runtime to compute its value in
    cases where the mask value isn't known until runtime.

    This pass resolves these calls into either 'true' or 'false' values so
    that later optimization passes can operate with these as constants.

    See stdlib.m4 for a number of uses of this idiom.
 */

class IsCompileTimeConstantPass : public llvm::FunctionPass {
  public:
    static char ID;
    IsCompileTimeConstantPass(bool last = false) : FunctionPass(ID) { isLastTry = last; }

    llvm::StringRef getPassName() const { return "Resolve \"is compile time constant\""; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);

    bool isLastTry;
};

char IsCompileTimeConstantPass::ID = 0;

bool IsCompileTimeConstantPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("IsCompileTimeConstantPass");

    llvm::Function *funcs[] = {m->module->getFunction("__is_compile_time_constant_mask"),
                               m->module->getFunction("__is_compile_time_constant_uniform_int32"),
                               m->module->getFunction("__is_compile_time_constant_varying_int32")};

    bool modifiedAny = false;
restart:
    for (llvm::BasicBlock::iterator i = bb.begin(), e = bb.end(); i != e; ++i) {
        // Iterate through the instructions looking for calls to the
        // __is_compile_time_constant_*() functions
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*i);
        if (callInst == NULL)
            continue;

        int j;
        int nFuncs = sizeof(funcs) / sizeof(funcs[0]);
        for (j = 0; j < nFuncs; ++j) {
            if (funcs[j] != NULL && callInst->getCalledFunction() == funcs[j])
                break;
        }
        if (j == nFuncs)
            // not a __is_compile_time_constant_* function
            continue;

        // This optimization pass can be disabled with both the (poorly
        // named) disableGatherScatterFlattening option and
        // disableMaskAllOnOptimizations.
        if (g->opt.disableGatherScatterFlattening || g->opt.disableMaskAllOnOptimizations) {
            llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, LLVMFalse);
            modifiedAny = true;
            goto restart;
        }

        // Is it a constant?  Bingo, turn the call's value into a constant
        // true value.
        llvm::Value *operand = callInst->getArgOperand(0);
        if (llvm::isa<llvm::Constant>(operand)) {
            llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, LLVMTrue);
            modifiedAny = true;
            goto restart;
        }

        // This pass runs multiple times during optimization.  Up until the
        // very last time, it only replaces the call with a 'true' if the
        // value is known to be constant and otherwise leaves the call
        // alone, in case further optimization passes can help resolve its
        // value.  The last time through, it eventually has to give up, and
        // replaces any remaining ones with 'false' constants.
        if (isLastTry) {
            llvm::ReplaceInstWithValue(i->getParent()->getInstList(), i, LLVMFalse);
            modifiedAny = true;
            goto restart;
        }
    }

    DEBUG_END_PASS("IsCompileTimeConstantPass");

    return modifiedAny;
}

bool IsCompileTimeConstantPass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("IsCompileTimeConstantPass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateIsCompileTimeConstantPass(bool isLastTry) { return new IsCompileTimeConstantPass(isLastTry); }

//////////////////////////////////////////////////////////////////////////
// DebugPass

/** This pass is added in list of passes after optimizations which
    we want to debug and print dump of LLVM IR in stderr. Also it
    prints name and number of previous optimization.
 */
#ifndef ISPC_NO_DUMPS
class DebugPass : public llvm::ModulePass {
  public:
    static char ID;
    DebugPass(char *output) : ModulePass(ID) { snprintf(str_output, sizeof(str_output), "%s", output); }

    llvm::StringRef getPassName() const { return "Dump LLVM IR"; }
    bool runOnModule(llvm::Module &m);

  private:
    char str_output[100];
};

char DebugPass::ID = 0;

bool DebugPass::runOnModule(llvm::Module &module) {
    fprintf(stderr, "%s", str_output);
    fflush(stderr);
    module.dump();
    return true;
}

static llvm::Pass *CreateDebugPass(char *output) { return new DebugPass(output); }
#endif

//////////////////////////////////////////////////////////////////////////
// DebugPassFile

/** This pass is added in list of passes after optimizations which
    we want to debug and print dump of LLVM IR to file.
 */
#ifndef ISPC_NO_DUMPS
class DebugPassFile : public llvm::ModulePass {
  public:
    static char ID;
    DebugPassFile(int number, llvm::StringRef name) : ModulePass(ID), pnum(number), pname(name) {}

    llvm::StringRef getPassName() const { return "Dump LLVM IR"; }
    bool runOnModule(llvm::Module &m);
    bool doInitialization(llvm::Module &m);

  private:
    void run(llvm::Module &m, bool init);
    int pnum;
    llvm::StringRef pname;
};

char DebugPassFile::ID = 0;

/**
 * Strips all non-alphanumeric characters from given string.
 */
std::string sanitize(std::string in) {
    llvm::Regex r("[^[:alnum:]]");
    while (r.match(in))
        in = r.sub("", in);
    return in;
}

void DebugPassFile::run(llvm::Module &module, bool init) {
    std::error_code EC;
    char fname[100];
    snprintf(fname, sizeof(fname), "%s_%d_%s.ll", init ? "init" : "ir", pnum, sanitize(std::string(pname)).c_str());
    llvm::raw_fd_ostream OS(fname, EC, llvm::sys::fs::OF_None);
    Assert(!EC && "IR dump file creation failed!");
    module.print(OS, 0);
}

bool DebugPassFile::runOnModule(llvm::Module &module) {
    run(module, false);
    return true;
}

bool DebugPassFile::doInitialization(llvm::Module &module) {
    run(module, true);
    return true;
}

static llvm::Pass *CreateDebugPassFile(int number, llvm::StringRef name) { return new DebugPassFile(number, name); }
#endif

///////////////////////////////////////////////////////////////////////////
// MakeInternalFuncsStaticPass

/** There are a number of target-specific functions that we use during
    these optimization passes.  By the time we are done with optimization,
    any uses of these should be inlined and no calls to these functions
    should remain.  This pass marks all of these functions as having
    private linkage so that subsequent passes can eliminate them as dead
    code, thus cleaning up the final code output by the compiler.  We can't
    just declare these as static from the start, however, since then they
    end up being eliminated as dead code during early optimization passes
    even though we may need to generate calls to them during later
    optimization passes.
 */
class MakeInternalFuncsStaticPass : public llvm::ModulePass {
  public:
    static char ID;
    MakeInternalFuncsStaticPass(bool last = false) : ModulePass(ID) {}

    void getAnalysisUsage(llvm::AnalysisUsage &AU) const { AU.setPreservesCFG(); }

    llvm::StringRef getPassName() const { return "Make internal funcs \"static\""; }
    bool runOnModule(llvm::Module &m);
};

char MakeInternalFuncsStaticPass::ID = 0;

bool MakeInternalFuncsStaticPass::runOnModule(llvm::Module &module) {
    const char *names[] = {
        "__avg_up_uint8",
        "__avg_up_int8",
        "__avg_up_uint16",
        "__avg_up_int16",
        "__avg_down_uint8",
        "__avg_down_int8",
        "__avg_down_uint16",
        "__avg_down_int16",
        "__fast_masked_vload",
        "__gather_factored_base_offsets32_i8",
        "__gather_factored_base_offsets32_i16",
        "__gather_factored_base_offsets32_i32",
        "__gather_factored_base_offsets32_i64",
        "__gather_factored_base_offsets32_half",
        "__gather_factored_base_offsets32_float",
        "__gather_factored_base_offsets32_double",
        "__gather_factored_base_offsets64_i8",
        "__gather_factored_base_offsets64_i16",
        "__gather_factored_base_offsets64_i32",
        "__gather_factored_base_offsets64_i64",
        "__gather_factored_base_offsets64_half",
        "__gather_factored_base_offsets64_float",
        "__gather_factored_base_offsets64_double",
        "__gather_base_offsets32_i8",
        "__gather_base_offsets32_i16",
        "__gather_base_offsets32_i32",
        "__gather_base_offsets32_i64",
        "__gather_base_offsets32_half",
        "__gather_base_offsets32_float",
        "__gather_base_offsets32_double",
        "__gather_base_offsets64_i8",
        "__gather_base_offsets64_i16",
        "__gather_base_offsets64_i32",
        "__gather_base_offsets64_i64",
        "__gather_base_offsets64_half",
        "__gather_base_offsets64_float",
        "__gather_base_offsets64_double",
        "__gather32_i8",
        "__gather32_i16",
        "__gather32_i32",
        "__gather32_i64",
        "__gather32_half",
        "__gather32_float",
        "__gather32_double",
        "__gather64_i8",
        "__gather64_i16",
        "__gather64_i32",
        "__gather64_i64",
        "__gather64_half",
        "__gather64_float",
        "__gather64_double",
        "__gather_elt32_i8",
        "__gather_elt32_i16",
        "__gather_elt32_i32",
        "__gather_elt32_i64",
        "__gather_elt32_half",
        "__gather_elt32_float",
        "__gather_elt32_double",
        "__gather_elt64_i8",
        "__gather_elt64_i16",
        "__gather_elt64_i32",
        "__gather_elt64_i64",
        "__gather_elt64_half",
        "__gather_elt64_float",
        "__gather_elt64_double",
        "__masked_load_i8",
        "__masked_load_i16",
        "__masked_load_i32",
        "__masked_load_i64",
        "__masked_load_half",
        "__masked_load_float",
        "__masked_load_double",
        "__masked_store_i8",
        "__masked_store_i16",
        "__masked_store_i32",
        "__masked_store_i64",
        "__masked_store_half",
        "__masked_store_float",
        "__masked_store_double",
        "__masked_store_blend_i8",
        "__masked_store_blend_i16",
        "__masked_store_blend_i32",
        "__masked_store_blend_i64",
        "__masked_store_blend_half",
        "__masked_store_blend_float",
        "__masked_store_blend_double",
        "__scatter_factored_base_offsets32_i8",
        "__scatter_factored_base_offsets32_i16",
        "__scatter_factored_base_offsets32_i32",
        "__scatter_factored_base_offsets32_i64",
        "__scatter_factored_base_offsets32_half",
        "__scatter_factored_base_offsets32_float",
        "__scatter_factored_base_offsets32_double",
        "__scatter_factored_base_offsets64_i8",
        "__scatter_factored_base_offsets64_i16",
        "__scatter_factored_base_offsets64_i32",
        "__scatter_factored_base_offsets64_i64",
        "__scatter_factored_base_offsets64_half",
        "__scatter_factored_base_offsets64_float",
        "__scatter_factored_base_offsets64_double",
        "__scatter_base_offsets32_i8",
        "__scatter_base_offsets32_i16",
        "__scatter_base_offsets32_i32",
        "__scatter_base_offsets32_i64",
        "__scatter_base_offsets32_half",
        "__scatter_base_offsets32_float",
        "__scatter_base_offsets32_double",
        "__scatter_base_offsets64_i8",
        "__scatter_base_offsets64_i16",
        "__scatter_base_offsets64_i32",
        "__scatter_base_offsets64_i64",
        "__scatter_base_offsets64_half",
        "__scatter_base_offsets64_float",
        "__scatter_base_offsets64_double",
        "__scatter_elt32_i8",
        "__scatter_elt32_i16",
        "__scatter_elt32_i32",
        "__scatter_elt32_i64",
        "__scatter_elt32_half",
        "__scatter_elt32_float",
        "__scatter_elt32_double",
        "__scatter_elt64_i8",
        "__scatter_elt64_i16",
        "__scatter_elt64_i32",
        "__scatter_elt64_i64",
        "__scatter_elt64_half",
        "__scatter_elt64_float",
        "__scatter_elt64_double",
        "__scatter32_i8",
        "__scatter32_i16",
        "__scatter32_i32",
        "__scatter32_i64",
        "__scatter32_half",
        "__scatter32_float",
        "__scatter32_double",
        "__scatter64_i8",
        "__scatter64_i16",
        "__scatter64_i32",
        "__scatter64_i64",
        "__scatter64_half",
        "__scatter64_float",
        "__scatter64_double",
        "__prefetch_read_varying_1",
        "__prefetch_read_varying_2",
        "__prefetch_read_varying_3",
        "__prefetch_read_varying_nt",
        "__prefetch_write_varying_1",
        "__prefetch_write_varying_2",
        "__prefetch_write_varying_3",
        "__keep_funcs_live",
#ifdef ISPC_XE_ENABLED
        "__masked_load_blend_i8",
        "__masked_load_blend_i16",
        "__masked_load_blend_i32",
        "__masked_load_blend_i64",
        "__masked_load_blend_half",
        "__masked_load_blend_float",
        "__masked_load_blend_double",
#endif
    };

    bool modifiedAny = false;
    int count = sizeof(names) / sizeof(names[0]);
    for (int i = 0; i < count; ++i) {
        llvm::Function *f = m->module->getFunction(names[i]);
        if (f != NULL && f->empty() == false) {
            f->setLinkage(llvm::GlobalValue::InternalLinkage);
            modifiedAny = true;
        }
    }

    return modifiedAny;
}

static llvm::Pass *CreateMakeInternalFuncsStaticPass() { return new MakeInternalFuncsStaticPass; }

///////////////////////////////////////////////////////////////////////////
// PeepholePass

class PeepholePass : public llvm::FunctionPass {
  public:
    PeepholePass();

    llvm::StringRef getPassName() const { return "Peephole Optimizations"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);

    static char ID;
};

char PeepholePass::ID = 0;

PeepholePass::PeepholePass() : FunctionPass(ID) {}

using namespace llvm::PatternMatch;

template <typename Op_t, unsigned Opcode> struct CastClassTypes_match {
    Op_t Op;
    const llvm::Type *fromType, *toType;

    CastClassTypes_match(const Op_t &OpMatch, const llvm::Type *f, const llvm::Type *t)
        : Op(OpMatch), fromType(f), toType(t) {}

    template <typename OpTy> bool match(OpTy *V) {
        if (llvm::Operator *O = llvm::dyn_cast<llvm::Operator>(V))
            return (O->getOpcode() == Opcode && Op.match(O->getOperand(0)) && O->getType() == toType &&
                    O->getOperand(0)->getType() == fromType);
        return false;
    }
};

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::SExt> m_SExt8To16(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::SExt>(Op, LLVMTypes::Int8VectorType,
                                                               LLVMTypes::Int16VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::ZExt> m_ZExt8To16(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::ZExt>(Op, LLVMTypes::Int8VectorType,
                                                               LLVMTypes::Int16VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::Trunc> m_Trunc16To8(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::Trunc>(Op, LLVMTypes::Int16VectorType,
                                                                LLVMTypes::Int8VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::SExt> m_SExt16To32(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::SExt>(Op, LLVMTypes::Int16VectorType,
                                                               LLVMTypes::Int32VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::ZExt> m_ZExt16To32(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::ZExt>(Op, LLVMTypes::Int16VectorType,
                                                               LLVMTypes::Int32VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::Trunc> m_Trunc32To16(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::Trunc>(Op, LLVMTypes::Int32VectorType,
                                                                LLVMTypes::Int16VectorType);
}

template <typename Op_t> struct UDiv2_match {
    Op_t Op;

    UDiv2_match(const Op_t &OpMatch) : Op(OpMatch) {}

    template <typename OpTy> bool match(OpTy *V) {
        llvm::BinaryOperator *bop;
        llvm::ConstantDataVector *cdv;
        if ((bop = llvm::dyn_cast<llvm::BinaryOperator>(V)) &&
            (cdv = llvm::dyn_cast<llvm::ConstantDataVector>(bop->getOperand(1))) && cdv->getSplatValue() != NULL) {
            const llvm::APInt &apInt = cdv->getUniqueInteger();

            switch (bop->getOpcode()) {
            case llvm::Instruction::UDiv:
                // divide by 2
                return (apInt.isIntN(2) && Op.match(bop->getOperand(0)));
            case llvm::Instruction::LShr:
                // shift left by 1
                return (apInt.isIntN(1) && Op.match(bop->getOperand(0)));
            default:
                return false;
            }
        }
        return false;
    }
};

template <typename V> inline UDiv2_match<V> m_UDiv2(const V &v) { return UDiv2_match<V>(v); }

template <typename Op_t> struct SDiv2_match {
    Op_t Op;

    SDiv2_match(const Op_t &OpMatch) : Op(OpMatch) {}

    template <typename OpTy> bool match(OpTy *V) {
        llvm::BinaryOperator *bop;
        llvm::ConstantDataVector *cdv;
        if ((bop = llvm::dyn_cast<llvm::BinaryOperator>(V)) &&
            (cdv = llvm::dyn_cast<llvm::ConstantDataVector>(bop->getOperand(1))) && cdv->getSplatValue() != NULL) {
            const llvm::APInt &apInt = cdv->getUniqueInteger();

            switch (bop->getOpcode()) {
            case llvm::Instruction::SDiv:
                // divide by 2
                return (apInt.isIntN(2) && Op.match(bop->getOperand(0)));
            case llvm::Instruction::AShr:
                // shift left by 1
                return (apInt.isIntN(1) && Op.match(bop->getOperand(0)));
            default:
                return false;
            }
        }
        return false;
    }
};

template <typename V> inline SDiv2_match<V> m_SDiv2(const V &v) { return SDiv2_match<V>(v); }

// Returns true if the given function has a call to an intrinsic function
// in its definition.
static bool lHasIntrinsicInDefinition(llvm::Function *func) {
    llvm::Function::iterator bbiter = func->begin();
    for (; bbiter != func->end(); ++bbiter) {
        for (llvm::BasicBlock::iterator institer = bbiter->begin(); institer != bbiter->end(); ++institer) {
            if (llvm::isa<llvm::IntrinsicInst>(institer))
                return true;
        }
    }
    return false;
}

static llvm::Instruction *lGetBinaryIntrinsic(const char *name, llvm::Value *opa, llvm::Value *opb) {
    llvm::Function *func = m->module->getFunction(name);
    Assert(func != NULL);

    // Make sure that the definition of the llvm::Function has a call to an
    // intrinsic function in its instructions; otherwise we will generate
    // infinite loops where we "helpfully" turn the default implementations
    // of target builtins like __avg_up_uint8 that are implemented with plain
    // arithmetic ops into recursive calls to themselves.
    if (lHasIntrinsicInDefinition(func))
        return lCallInst(func, opa, opb, name);
    else
        return NULL;
}

//////////////////////////////////////////////////

static llvm::Instruction *lMatchAvgUpUInt8(llvm::Value *inst) {
    // (unsigned int8)(((unsigned int16)a + (unsigned int16)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst, m_Trunc16To8(m_UDiv2(m_CombineOr(
                        m_CombineOr(m_Add(m_ZExt8To16(m_Value(opa)), m_Add(m_ZExt8To16(m_Value(opb)), m_APInt(delta))),
                                    m_Add(m_Add(m_ZExt8To16(m_Value(opa)), m_APInt(delta)), m_ZExt8To16(m_Value(opb)))),
                        m_Add(m_Add(m_ZExt8To16(m_Value(opa)), m_ZExt8To16(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return NULL;

        return lGetBinaryIntrinsic("__avg_up_uint8", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgDownUInt8(llvm::Value *inst) {
    // (unsigned int8)(((unsigned int16)a + (unsigned int16)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc16To8(m_UDiv2(m_Add(m_ZExt8To16(m_Value(opa)), m_ZExt8To16(m_Value(opb))))))) {
        return lGetBinaryIntrinsic("__avg_down_uint8", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgUpUInt16(llvm::Value *inst) {
    // (unsigned int16)(((unsigned int32)a + (unsigned int32)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst,
              m_Trunc32To16(m_UDiv2(m_CombineOr(
                  m_CombineOr(m_Add(m_ZExt16To32(m_Value(opa)), m_Add(m_ZExt16To32(m_Value(opb)), m_APInt(delta))),
                              m_Add(m_Add(m_ZExt16To32(m_Value(opa)), m_APInt(delta)), m_ZExt16To32(m_Value(opb)))),
                  m_Add(m_Add(m_ZExt16To32(m_Value(opa)), m_ZExt16To32(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return NULL;

        return lGetBinaryIntrinsic("__avg_up_uint16", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgDownUInt16(llvm::Value *inst) {
    // (unsigned int16)(((unsigned int32)a + (unsigned int32)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc32To16(m_UDiv2(m_Add(m_ZExt16To32(m_Value(opa)), m_ZExt16To32(m_Value(opb))))))) {
        return lGetBinaryIntrinsic("__avg_down_uint16", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgUpInt8(llvm::Value *inst) {
    // (int8)(((int16)a + (int16)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst, m_Trunc16To8(m_SDiv2(m_CombineOr(
                        m_CombineOr(m_Add(m_SExt8To16(m_Value(opa)), m_Add(m_SExt8To16(m_Value(opb)), m_APInt(delta))),
                                    m_Add(m_Add(m_SExt8To16(m_Value(opa)), m_APInt(delta)), m_SExt8To16(m_Value(opb)))),
                        m_Add(m_Add(m_SExt8To16(m_Value(opa)), m_SExt8To16(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return NULL;

        return lGetBinaryIntrinsic("__avg_up_int8", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgDownInt8(llvm::Value *inst) {
    // (int8)(((int16)a + (int16)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc16To8(m_SDiv2(m_Add(m_SExt8To16(m_Value(opa)), m_SExt8To16(m_Value(opb))))))) {
        return lGetBinaryIntrinsic("__avg_down_int8", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgUpInt16(llvm::Value *inst) {
    // (int16)(((int32)a + (int32)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst,
              m_Trunc32To16(m_SDiv2(m_CombineOr(
                  m_CombineOr(m_Add(m_SExt16To32(m_Value(opa)), m_Add(m_SExt16To32(m_Value(opb)), m_APInt(delta))),
                              m_Add(m_Add(m_SExt16To32(m_Value(opa)), m_APInt(delta)), m_SExt16To32(m_Value(opb)))),
                  m_Add(m_Add(m_SExt16To32(m_Value(opa)), m_SExt16To32(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return NULL;

        return lGetBinaryIntrinsic("__avg_up_int16", opa, opb);
    }
    return NULL;
}

static llvm::Instruction *lMatchAvgDownInt16(llvm::Value *inst) {
    // (int16)(((int32)a + (int32)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc32To16(m_SDiv2(m_Add(m_SExt16To32(m_Value(opa)), m_SExt16To32(m_Value(opb))))))) {
        return lGetBinaryIntrinsic("__avg_down_int16", opa, opb);
    }
    return NULL;
}

bool PeepholePass::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("PeepholePass");

    bool modifiedAny = false;
restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::Instruction *inst = &*iter;

        llvm::Instruction *builtinCall = lMatchAvgUpUInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgUpUInt16(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownUInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownUInt16(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgUpInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgUpInt16(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownInt16(inst);
        if (builtinCall != NULL) {
            llvm::ReplaceInstWithInst(inst, builtinCall);
            modifiedAny = true;
            goto restart;
        }
    }

    DEBUG_END_PASS("PeepholePass");

    return modifiedAny;
}

bool PeepholePass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("PeepholePass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreatePeepholePass() { return new PeepholePass; }

/** Given an llvm::Value known to be an integer, return its value as
    an int64_t.
*/
static int64_t lGetIntValue(llvm::Value *offset) {
    llvm::ConstantInt *intOffset = llvm::dyn_cast<llvm::ConstantInt>(offset);
    Assert(intOffset && (intOffset->getBitWidth() == 32 || intOffset->getBitWidth() == 64));
    return intOffset->getSExtValue();
}

///////////////////////////////////////////////////////////////////////////
// ReplaceStdlibShiftPass

class ReplaceStdlibShiftPass : public llvm::FunctionPass {
  public:
    static char ID;
    ReplaceStdlibShiftPass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Resolve \"replace extract insert chains\""; }

    bool runOnBasicBlock(llvm::BasicBlock &BB);

    bool runOnFunction(llvm::Function &F);
};

char ReplaceStdlibShiftPass::ID = 0;

// This pass replaces shift() with ShuffleVector when the offset is a constant.
// rotate() which is similar in functionality has a slightly different
// implementation. This is due to LLVM(createInstructionCombiningPass)
// optimizing rotate() implementation better when similar implementations
// are used for both. This is a hack to produce similarly optimized code for
// shift.
bool ReplaceStdlibShiftPass::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("ReplaceStdlibShiftPass");
    bool modifiedAny = false;

    llvm::Function *shifts[6];
    shifts[0] = m->module->getFunction("shift___vytuni");
    shifts[1] = m->module->getFunction("shift___vysuni");
    shifts[2] = m->module->getFunction("shift___vyiuni");
    shifts[3] = m->module->getFunction("shift___vyIuni");
    shifts[4] = m->module->getFunction("shift___vyfuni");
    shifts[5] = m->module->getFunction("shift___vyduni");

    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::Instruction *inst = &*iter;

        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            llvm::Function *func = ci->getCalledFunction();
            for (int i = 0; i < 6; i++) {
                if (shifts[i] && (shifts[i] == func)) {
                    // we matched a call
                    llvm::Value *shiftedVec = ci->getArgOperand(0);
                    llvm::Value *shiftAmt = ci->getArgOperand(1);
                    if (llvm::isa<llvm::Constant>(shiftAmt)) {
                        int vectorWidth = g->target->getVectorWidth();
                        int *shuffleVals = new int[vectorWidth];
                        int shiftInt = lGetIntValue(shiftAmt);
                        for (int i = 0; i < vectorWidth; i++) {
                            int s = i + shiftInt;
                            s = (s < 0) ? vectorWidth : s;
                            s = (s >= vectorWidth) ? vectorWidth : s;
                            shuffleVals[i] = s;
                        }
                        llvm::Value *shuffleIdxs = LLVMInt32Vector(shuffleVals);
                        llvm::Value *zeroVec = llvm::ConstantAggregateZero::get(shiftedVec->getType());
                        llvm::Value *shuffle =
                            new llvm::ShuffleVectorInst(shiftedVec, zeroVec, shuffleIdxs, "vecShift", ci);
                        ci->replaceAllUsesWith(shuffle);
                        modifiedAny = true;
                        delete[] shuffleVals;
                    } else if (g->opt.level > 0) {
                        PerformanceWarning(SourcePos(), "Stdlib shift() called without constant shift amount.");
                    }
                }
            }
        }
    }

    DEBUG_END_PASS("ReplaceStdlibShiftPass");

    return modifiedAny;
}

bool ReplaceStdlibShiftPass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("ReplaceStdlibShiftPass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateReplaceStdlibShiftPass() { return new ReplaceStdlibShiftPass(); }

#ifdef ISPC_XE_ENABLED

///////////////////////////////////////////////////////////////////////////
// Xe Coalescing passes

/*
 * MemoryCoalescing: basis for store/load coalescing optimizations.
 *
 * Memory coalescing tries to merge several memory operations into
 * wider ones. There are two different flows for store and load coalescings.
 *
 * This is a function pass. Each BB is handled separately.
 *
 * The general idea of optimization is to iterate over BB and analyse pointers
 * for all optimization targets (target is an instruction that can be optimized).
 * Instructions that uses the same pointer can be coalesced.
 *
 * Each memory instruction has a pointer operand. To perform coalescing,
 * we need to determine its base pointer and offset for it. This is done
 * via recurrent algorithm. Only constant offset can be handled.
 * TODO: found in experiments: sometimes pointer is passed as null
 * and actual pointer came via calculated offset. Such situation is not
 * handled now (Target: Xe-TPM).
 *
 * Sometimies coalescing can fail in case when it is not sure if the
 * transformation is actually safe. This can happen if some "bad"
 * instructions met between optimization targets. To handle this, all
 * optimization targets are collected in several blocks. They also can be
 * treated as non-interfering fragments of BB. Each block can be
 * safely optimized. Bad instructions trigger creation of such blocks.
 * For now, stopper detection is implemented in a general way for all
 * types of coalescing. It can be changed in future.
 *
 * Basicly, implementation can treat offset in different ways. Afterall,
 * implementation builds accesses for optimized instructions. All
 * offsets are presented in bytes now.
 *
 * To create new coalescing type, one should implement several handlers.
 * More info about them below.
 *
 * There are also some helpers that are not used by optimization
 * directly, but can be used by several coalescing implementations.
 * Such helpers contains some general stuff.
 */
class MemoryCoalescing : public llvm::FunctionPass {
  protected:
    typedef int64_t OffsetT;
    typedef llvm::SmallVector<OffsetT, 16> OffsetsVecT;
    typedef llvm::SmallVector<llvm::Instruction *, 4> BlockInstsVecT;

    // Available coalescing types
    enum class MemType { OPT_LOAD, OPT_STORE };

    // Inst optimization data
    class InstData {
      public:
        // Instruction itself
        llvm::Instruction *Inst;
        // Offsets for memory accesses
        OffsetsVecT Offsets;
        // Unused for loads, stored value for stores
        llvm::Value *Val;
        InstData() = delete;
        InstData(llvm::Instruction *Inst, OffsetsVecT &Offsets, llvm::Value *Val)
            : Inst(Inst), Offsets(Offsets), Val(Val) {}
    };

    // Ptr optimization data
    class PtrData {
      public:
        // The insertion point for the ptr users.
        // All instructions will be placed before it.
        // InsertPoint should be the first instruction in block for load coalescing
        // and the last one for store coalescing. This is achieved by different
        // traversal in both types of optimization.
        llvm::Instruction *InsertPoint;
        // Data for insts that use this ptr
        std::vector<InstData> InstsData;
        PtrData() : InsertPoint(nullptr) {}
        void addInstruction(llvm::Instruction *Inst, OffsetsVecT &Offsets, llvm::Value *Val) {
            InstsData.push_back(InstData(Inst, Offsets, Val));
            if (!InsertPoint)
                InsertPoint = Inst;
        }
    };

    // Data for optimization block
    class OptBlock {
      public:
        // Data for all ptrs in the block
        std::unordered_map<llvm::Value *, PtrData> PtrsData;
        void addInstruction(llvm::Instruction *Inst, llvm::Value *Ptr, OffsetsVecT &Offsets,
                            llvm::Value *Val = nullptr) {
            PtrsData[Ptr].addInstruction(Inst, Offsets, Val);
        }
    };

    // Instructions that are possibly dead. It includes some newly created instructions that
    // are created during optimization and memory instructions that are optimized out. All
    // instructions from this list are treated as dead if they don't have any users.
    std::set<llvm::Instruction *> PossiblyDead;
    // Modification flag
    bool modifiedAny = false;
    // Address space for optimization
    // TODO: not obvious how it would work with TPM on Xe now.
    AddressSpace AddrSpace;

  private:
    // Info for complex GEPs
    struct GEPVarOffsetInfo {
        llvm::Instruction *GEP;
        llvm::User::value_op_iterator FirstConstUse;
        GEPVarOffsetInfo(llvm::Instruction *GEP) : GEP(GEP) {}
    };

    // DenseMap helper for complex GEPs
    struct DenseMapInfo {
        static inline GEPVarOffsetInfo getEmptyKey() {
            return GEPVarOffsetInfo(llvm::DenseMapInfo<llvm::Instruction *>::getEmptyKey());
        }
        static inline GEPVarOffsetInfo getTombstoneKey() {
            return GEPVarOffsetInfo(llvm::DenseMapInfo<llvm::Instruction *>::getTombstoneKey());
        }
        static inline bool isSentinel(const GEPVarOffsetInfo &Val) {
            return Val.GEP == getEmptyKey().GEP || Val.GEP == getTombstoneKey().GEP;
        }
        static unsigned getHashValue(const GEPVarOffsetInfo &Val) {
            return hash_combine_range(Val.GEP->value_op_begin(), Val.FirstConstUse);
        }
        static bool isEqual(const GEPVarOffsetInfo &LHS, const GEPVarOffsetInfo &RHS) {
            if (isSentinel(LHS) || isSentinel(RHS))
                return LHS.GEP == RHS.GEP;

            for (auto lhs_it = LHS.GEP->value_op_begin(), rhs_it = RHS.GEP->value_op_begin(), lhs_e = LHS.FirstConstUse,
                      rhs_e = RHS.FirstConstUse;
                 lhs_it != lhs_e || rhs_it != rhs_e; ++lhs_it, ++rhs_it) {
                if (lhs_it == lhs_e || rhs_it == rhs_e)
                    return false;
                if (*lhs_it != *rhs_it)
                    return false;
            }

            return true;
        }
    };

    // Helper for ptr offset analysis. It holds
    // Ptr+Offset data. IsConstantOffset is used to
    // check if optimization is appliable for such ptr.
    // Ptr can be null, in this case only Offset
    // matters. This is used for arithmetic analysis.
    // The data is consistent only if the IsConstantOffset
    // flag is true.
    struct BasePtrInfo {
        llvm::Value *Ptr;
        OffsetT Offset;
        bool IsConstantOffset;

        BasePtrInfo() : Ptr(nullptr), Offset(0), IsConstantOffset(false) {}
    };

    // Coalescing type
    const MemType OptType;
    // Cached data for visited ptr instructions
    std::unordered_map<llvm::Value *, BasePtrInfo> BasePtrInfoCache;
    // Cached data for complex GEPs partial replacements
    llvm::DenseMap<GEPVarOffsetInfo, llvm::Value *, DenseMapInfo> ComplexGEPsInfoCache;
    // Blocks to be optimized
    std::vector<OptBlock> Blocks;

    // Find base ptr and its offset
    BasePtrInfo findBasePtr(llvm::Value *PtrOperand);
    // Analyse GEP that has variable offset. This is used by findBasePtr.
    // Such GEPs can have same variable part while the final (different) part is constant.
    BasePtrInfo analyseVarOffsetGEP(llvm::GetElementPtrInst *GEP);
    // Analyse arithmetics. That allows to handle more cases when offset is calculated not via GEP.
    // TODO: not implemented now, returns result with IsConstantOffset=false.
    BasePtrInfo analyseArithmetics(llvm::BinaryOperator *Arithm);
    // Return true if Inst blocks further optimization of currently collected
    // optimization targets. This is a stopper for collecting
    // instructions.
    bool stopsOptimization(llvm::Instruction *Inst) const;
    // Add Block to blocks list and return new one.
    OptBlock finishBlock(OptBlock &Block);
    // Collect data for optimization
    void analyseInsts(llvm::BasicBlock &BB);
    // Apply coalescing
    void applyOptimization();
    // Delete dead instructions
    void deletePossiblyDeadInsts();
    // Reset all internal structures
    void clear();

  protected:
    // Initialization
    MemoryCoalescing(char &ID, MemType OptType, AddressSpace AddrSpace)
        : FunctionPass(ID), AddrSpace(AddrSpace), OptType(OptType) {}
    // Optimization runner
    bool runOnFunction(llvm::Function &Fn);

    /* ------ Handlers ------ */
    // Methods in this block are interface for different coalescing types.

    // Return true if coalescing can handle Inst.
    virtual bool isOptimizationTarget(llvm::Instruction *Inst) const = 0;
    // Return pointer value or null if there is no one. This should handle
    // all optimization targets.
    virtual llvm::Value *getPointer(llvm::Instruction *Inst) const = 0;
    // Return offset for implied values. Scatter and gathers, for example,
    // can have vectorized offset, so the result is a list. If the offset
    // is not constant, return empty list. This should handle all optimization
    // targets.
    virtual OffsetsVecT getOffset(llvm::Instruction *Inst) const = 0;
    // Return value being stored. For load coalescing, simply return null.
    // This function is not called under load coalescing. For store coalescing,
    // this should handle all optimizations target.
    virtual llvm::Value *getStoredValue(llvm::Instruction *Inst) const = 0;
    // Perform optimization on ptr data.
    virtual void optimizePtr(llvm::Value *Ptr, PtrData &PD, llvm::Instruction *InsertPoint) = 0;

    // TODO: this handler must call runOnBasicBlockImpl to run optimization.
    // This function is needed due to DEBUG_START/END_PASS logic. Maybe there is
    // a better way to solve it.
    virtual void runOnBasicBlock(llvm::BasicBlock &BB) = 0;
    void runOnBasicBlockImpl(llvm::BasicBlock &BB);

    /* ------- Helpers ------ */
    // Methods in this block are not used in optimization directly
    // and can be invoked by handlers implementations

    // Collect constant offsets from vector. If offset is not a constant, empty list
    // is returned.
    OffsetsVecT getConstOffsetFromVector(llvm::Value *VecVal) const;
    // Multiplies all elements by scale.
    void applyScale(OffsetsVecT &Offsets, OffsetT scale) const;
    // Build InsertElementInst. This is offset based insertion and it can deal with mixed types.
    llvm::Value *buildIEI(llvm::Value *InsertTo, llvm::Value *Val, OffsetT OffsetBytes,
                          llvm::Instruction *InsertBefore) const;
    // Build ExtractElementInst. This is offset based extraction and it can deal with mixed types.
    llvm::Value *buildEEI(llvm::Value *ExtractFrom, OffsetT OffsetBytes, llvm::Type *DstTy,
                          llvm::Instruction *InsertBefore) const;
    // Build Cast (BitCast or IntToPtr)
    llvm::Value *buildCast(llvm::Value *Val, llvm::Type *DstTy, llvm::Instruction *InsertBefore) const;
    // Extract element from block values. Can aggregate value from several block instructions.
    llvm::Value *extractValueFromBlock(const BlockInstsVecT &BlockInstsVec, OffsetT OffsetBytes, llvm::Type *DstTy,
                                       llvm::Instruction *InsertBefore) const;
    // Get scalar type size in bytes
    unsigned getScalarTypeSize(llvm::Type *Ty) const;
};

// Optimization runner
bool MemoryCoalescing::runOnFunction(llvm::Function &Fn) {
    llvm::TimeTraceScope FuncScope("MemoryCoalescing::runOnFunction", Fn.getName());
    for (llvm::BasicBlock &BB : Fn) {
        runOnBasicBlock(BB);
    }

    return modifiedAny;
}

// Find base pointer info for ptr operand.
MemoryCoalescing::BasePtrInfo MemoryCoalescing::findBasePtr(llvm::Value *PtrOperand) {
    // Look for previously handled value
    auto it = BasePtrInfoCache.find(PtrOperand);
    if (it != BasePtrInfoCache.end())
        return it->second;

    BasePtrInfo res;
    if (auto BCI = llvm::dyn_cast<llvm::BitCastInst>(PtrOperand)) {
        res = findBasePtr(BCI->getOperand(0));
    } else if (auto GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(PtrOperand)) {
        if (GEP->hasAllConstantIndices()) {
            // Easy case. Collect offset.
            llvm::APInt acc(g->target->is32Bit() ? 32 : 64, 0, true);
            bool checker = GEP->accumulateConstantOffset(GEP->getModule()->getDataLayout(), acc);
            Assert(checker);

            // Run this analysis for GEP's ptr operand to handle possible bypasses
            res = findBasePtr(GEP->getPointerOperand());
            res.Offset += acc.getSExtValue();
        } else {
            // Bad case. We need partialy use the data in this GEP - there might be
            // intersections with others. The handler for this case is implemented
            // in a separate method.
            res = analyseVarOffsetGEP(GEP);
        }
    } else if (auto Arithm = llvm::dyn_cast<llvm::BinaryOperator>(PtrOperand)) {
        // Handle arithmetics
        res = analyseArithmetics(Arithm);
    } else if (llvm::isa<llvm::PointerType>(PtrOperand->getType())) {
        // This block is reached when no possible bypasses are left.
        // Use this ptr as a base
        res.Ptr = PtrOperand;
        res.Offset = 0;
        res.IsConstantOffset = true;
    } else if (auto Const = llvm::dyn_cast<llvm::ConstantInt>(PtrOperand)) {
        // Constant operand - set offset.
        res.Ptr = nullptr;
        res.Offset = Const->getSExtValue();
        res.IsConstantOffset = true;
    } else {
        // That is a non-constant offset
        res.IsConstantOffset = false;
    }

    // Finally, cache result and return
    BasePtrInfoCache[PtrOperand] = res;
    return res;
}

// Analyse GEP that has some indices non-constant.
// TODO: Consider adding full chain analysis. Decided to leave this idea for now
// because it may introduce many redundant address calculations: current approach
// would copy full chain even if some common part could be calculated somewhere eariler.
MemoryCoalescing::BasePtrInfo MemoryCoalescing::analyseVarOffsetGEP(llvm::GetElementPtrInst *GEP) {
    BasePtrInfo res;
    // Find the last constant idxs: we may be able to use them as constant offset for
    // several GEPs with common pre-constant part
    auto FirstConstIdx = GEP->getNumOperands();
    for (unsigned i = FirstConstIdx - 1; i >= 0; --i) {
        if (!llvm::isa<llvm::ConstantInt>(GEP->getOperand(i)))
            break;
        FirstConstIdx = i;
    }
    // Early return in case when no constant offset was found:
    // further actions are useless.
    if (GEP->getNumOperands() == FirstConstIdx) {
        res.Ptr = GEP;
        res.Offset = 0;
        res.IsConstantOffset = true;
        return res;
    }
    // Initialize var offset ops info.
    auto FirstConstUse = GEP->value_op_begin();
    for (unsigned i = 0; i < FirstConstIdx; ++i)
        FirstConstUse++;
    GEPVarOffsetInfo GEPVarOffsetData(GEP);
    GEPVarOffsetData.FirstConstUse = FirstConstUse;
    // Try to find existing dangling PartialGEP for such Ops combination
    // or create new one.
    auto DanglingGEP_it = ComplexGEPsInfoCache.find(GEPVarOffsetData);
    if (DanglingGEP_it == ComplexGEPsInfoCache.end()) {
        std::vector<llvm::Value *> PartialIdxs;
        for (unsigned i = 1; i < FirstConstIdx; ++i)
            PartialIdxs.push_back(GEP->getOperand(i));
        llvm::Value *tPtr = GEP->getPointerOperand();
        llvm::Type *tType = llvm::cast<llvm::PointerType>(tPtr->getType()->getScalarType())->getElementType();
        auto ret = ComplexGEPsInfoCache.insert(
            {GEPVarOffsetData, llvm::GetElementPtrInst::Create(tType, tPtr, PartialIdxs, "partial_gep")});
        DanglingGEP_it = ret.first;
    }
    llvm::Value *DanglingGEP = DanglingGEP_it->second;
    // Collect idxs. We will use them to find offset. Push zero constant first:
    // this will be needed for correct offset accumulation further.
    std::vector<llvm::Value *> Idxs = {llvm::ConstantInt::get(LLVMTypes::Int32Type, 0)};
    for (unsigned i = FirstConstIdx; i < GEP->getNumOperands(); ++i) {
        Idxs.push_back(GEP->getOperand(i));
    }
    // Get partial GEP type
    llvm::PointerType *PartialType = llvm::cast<llvm::PointerType>(DanglingGEP->getType());
    // Create temporary GEP that will help us to get some useful info
    llvm::Value *tPtr = llvm::ConstantPointerNull::get(PartialType);
    llvm::Type *tType = llvm::cast<llvm::PointerType>(tPtr->getType()->getScalarType())->getElementType();
    llvm::GetElementPtrInst *GEPHelper = llvm::GetElementPtrInst::Create(tType, tPtr, Idxs);
    // Accumulate offset from helper
    llvm::APInt acc(g->target->is32Bit() ? 32 : 64, 0, true);
    bool checker = GEPHelper->accumulateConstantOffset(GEP->getModule()->getDataLayout(), acc);
    Assert(checker);
    // Finally, store data.
    res.Ptr = DanglingGEP;
    res.Offset = acc.getSExtValue();
    res.IsConstantOffset = true;

    return res;
}

// Analyse arithmetic calculations on pointer.
// TODO: not implemented, returns stopper
MemoryCoalescing::BasePtrInfo MemoryCoalescing::analyseArithmetics(llvm::BinaryOperator *Arithm) {
    BasePtrInfo res;
    res.IsConstantOffset = false;
    return res;
}

// Basic block optimization runner.
// TODO: runOnBasicBlock must call it to run optimization. See comment above.
void MemoryCoalescing::runOnBasicBlockImpl(llvm::BasicBlock &BB) {
    analyseInsts(BB);
    applyOptimization();
    deletePossiblyDeadInsts();
    clear();
}

void MemoryCoalescing::deletePossiblyDeadInsts() {
    for (auto *Inst : PossiblyDead) {
        if (Inst->use_empty())
            Inst->eraseFromParent();
    }
}

void MemoryCoalescing::clear() {
    BasePtrInfoCache.clear();
    ComplexGEPsInfoCache.clear();
    PossiblyDead.clear();
    Blocks.clear();
}

// Analyse instructions in BB. Gather them into optimizable blocks.
void MemoryCoalescing::analyseInsts(llvm::BasicBlock &BB) {
    auto bi = BB.begin(), be = BB.end();
    auto rbi = BB.rbegin(), rbe = BB.rend();
    Assert(OptType == MemType::OPT_LOAD || OptType == MemType::OPT_STORE);
    OptBlock CurrentBlock;
    for (; (OptType == MemType::OPT_LOAD) ? (bi != be) : (rbi != rbe);) {
        llvm::Instruction *Inst = NULL;
        if (OptType == MemType::OPT_LOAD) {
            Inst = &*bi;
            ++bi;
        } else {
            Inst = &*rbi;
            ++rbi;
        }

        if (isOptimizationTarget(Inst)) {
            // Find ptr and offsets
            BasePtrInfo BasePtr = findBasePtr(getPointer(Inst));
            OffsetsVecT Offsets = getOffset(Inst);
            if (BasePtr.IsConstantOffset && BasePtr.Ptr && !Offsets.empty()) {
                if (OptType == MemType::OPT_STORE && !CurrentBlock.PtrsData.empty()) {
                    Assert(CurrentBlock.PtrsData.size() == 1 && "Store coalescing can handle only one pointer at once");
                    if (CurrentBlock.PtrsData.find(BasePtr.Ptr) == CurrentBlock.PtrsData.end()) {
                        // Finish current block so the instruction is added to the new one
                        CurrentBlock = finishBlock(CurrentBlock);
                    }
                }

                // Recalculate offsets for BasePtr
                for (auto &Offset : Offsets)
                    Offset += BasePtr.Offset;
                // Add inst to block
                CurrentBlock.addInstruction(Inst, BasePtr.Ptr, Offsets,
                                            OptType == MemType::OPT_STORE ? getStoredValue(Inst) : nullptr);
                // Instruction that was added to the block won't block optimization
                continue;
            }
        }

        if (stopsOptimization(Inst)) {
            // Add current block and create new one
            CurrentBlock = finishBlock(CurrentBlock);
        }
    }

    // Add current block
    Blocks.push_back(CurrentBlock);
}

// Apply optimization for all optimizable blocks.
void MemoryCoalescing::applyOptimization() {
    for (auto &Block : Blocks) {
        for (auto &PD : Block.PtrsData) {
            // Apply dangling GEP
            if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(PD.first))
                if (!GEP->getParent())
                    GEP->insertBefore(PD.second.InsertPoint);
            // Run optimization
            optimizePtr(PD.first, PD.second, PD.second.InsertPoint);
        }
    }
}

// Check whether instruction blocks optimization.
// Optimization target checker should be called before because any store
// blocks optimization in general case.
bool MemoryCoalescing::stopsOptimization(llvm::Instruction *Inst) const {
    if (OptType == MemType::OPT_LOAD) {
        // For load coalescing, only stores can introduce problems.
        if (auto CI = llvm::dyn_cast<llvm::CallInst>(Inst)) {
            // Such call may introduce load-store-load sequence
            if (!CI->onlyReadsMemory())
                for (unsigned i = 0; i < Inst->getNumOperands(); ++i)
                    if (GetAddressSpace(Inst->getOperand(i)) == AddrSpace)
                        return true;
        } else if (auto SI = llvm::dyn_cast<llvm::StoreInst>(Inst)) {
            // May introduce load-store-load sequence
            if (GetAddressSpace(SI->getPointerOperand()) == AddrSpace)
                return true;
        }
    } else if (OptType == MemType::OPT_STORE) {
        // For store coalescing, both loads and stores can introduce problems.
        if (auto CI = llvm::dyn_cast<llvm::CallInst>(Inst)) {
            // Such call may introduce store-load/store-store sequence
            for (unsigned i = 0; i < Inst->getNumOperands(); ++i) {
                if (GetAddressSpace(Inst->getOperand(i)) == AddrSpace)
                    return true;
            }
        } else if (auto LI = llvm::dyn_cast<llvm::LoadInst>(Inst)) {
            // May introduce store-load-store sequence
            if (GetAddressSpace(LI->getPointerOperand()) == AddrSpace)
                return true;
        } else if (auto SI = llvm::dyn_cast<llvm::StoreInst>(Inst)) {
            // May introduce store-store-store sequence
            if (GetAddressSpace(SI->getPointerOperand()) == AddrSpace)
                return true;
        }
    } else
        Assert(0 && "Bad optimization type");

    return false;
}

// Store current block and init new one.
MemoryCoalescing::OptBlock MemoryCoalescing::finishBlock(MemoryCoalescing::OptBlock &Block) {
    Blocks.push_back(Block);
    return OptBlock();
}

// Get constant offset from vector.
MemoryCoalescing::OffsetsVecT MemoryCoalescing::getConstOffsetFromVector(llvm::Value *VecVal) const {
    Assert(VecVal && llvm::isa<llvm::VectorType>(VecVal->getType()) && "Expected vector type");
    OffsetsVecT res;
    auto ConstVec = llvm::dyn_cast<llvm::ConstantDataVector>(VecVal);

    if (!ConstVec)
        return res;
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    for (unsigned i = 0, size = llvm::cast<llvm::FixedVectorType>(ConstVec->getType())->getNumElements(); i < size;
         ++i) {
#else
    for (unsigned i = 0, size = llvm::cast<llvm::VectorType>(ConstVec->getType())->getNumElements(); i < size; ++i) {
#endif
        auto ConstantInt = llvm::dyn_cast<llvm::ConstantInt>(ConstVec->getElementAsConstant(i));
        if (!ConstantInt) {
            // Actually, we don't expect this to happen
            res.clear();
            break;
        }
        res.push_back(ConstantInt->getSExtValue());
    }

    return res;
}

// Apply scale on offsets.
void MemoryCoalescing::applyScale(MemoryCoalescing::OffsetsVecT &Offsets, MemoryCoalescing::OffsetT Scale) const {
    for (auto &Offset : Offsets)
        Offset *= Scale;
}

unsigned MemoryCoalescing::getScalarTypeSize(llvm::Type *Ty) const {
    Ty = Ty->getScalarType();
    if (Ty->isPointerTy())
        return g->target->is32Bit() ? 4 : 8;
    return Ty->getPrimitiveSizeInBits() >> 3;
}

llvm::Value *MemoryCoalescing::buildIEI(llvm::Value *InsertTo, llvm::Value *Val, MemoryCoalescing::OffsetT OffsetBytes,
                                        llvm::Instruction *InsertBefore) const {
    llvm::Type *ScalarType = InsertTo->getType()->getScalarType();
    llvm::Type *ValTy = Val->getType();

    Assert(!ValTy->isVectorTy() && !ValTy->isAggregateType() && "Expected scalar type");
    Assert(InsertTo->getType()->isVectorTy() && "Expected vector type");

    unsigned ScalarTypeBytes = getScalarTypeSize(ScalarType);
    unsigned ValTyBytes = getScalarTypeSize(ValTy);
    unsigned Idx = OffsetBytes / ScalarTypeBytes;
    unsigned Rem = OffsetBytes % ScalarTypeBytes;
    Idx = Rem ? Idx + 1 : Idx;
    llvm::Value *FinalInsertElement = Val;
    if (ValTy != ScalarType) {
        if (ValTyBytes == ScalarTypeBytes) {
            // Apply cast
            FinalInsertElement = buildCast(Val, ScalarType, InsertBefore);
        } else {
            // Need to create eei-cast-iei-cast chain.
            // Extract scalar type value
            auto *EEI = llvm::ExtractElementInst::Create(InsertTo, llvm::ConstantInt::get(LLVMTypes::Int64Type, Idx),
                                                         "mem_coal_diff_ty_eei", InsertBefore);
            // Cast it to vector of smaller types
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
            auto *Cast = buildCast(EEI, llvm::FixedVectorType::get(ValTy, ScalarTypeBytes / ValTyBytes), InsertBefore);
#else
            auto *Cast = buildCast(EEI, llvm::VectorType::get(ValTy, ScalarTypeBytes / ValTyBytes), InsertBefore);
#endif
            // Insert value into casted type. Do it via this builder so we don't duplicate logic of offset calculations.
            auto *IEI = buildIEI(Cast, Val, Rem, InsertBefore);
            // Cast to original type
            FinalInsertElement = buildCast(IEI, ScalarType, InsertBefore);
        }
    }

    return llvm::InsertElementInst::Create(
        InsertTo, FinalInsertElement, llvm::ConstantInt::get(LLVMTypes::Int64Type, Idx), "mem_coal_iei", InsertBefore);
}

llvm::Value *MemoryCoalescing::buildCast(llvm::Value *Val, llvm::Type *DstTy, llvm::Instruction *InsertBefore) const {
    // No cast needed: early return
    if (Val->getType() == DstTy)
        return Val;

    if (DstTy->isPointerTy() && !Val->getType()->isPointerTy()) {
        return new llvm::IntToPtrInst(Val, DstTy, "coal_diff_ty_ptr_cast", InsertBefore);
    } else if (!DstTy->isPointerTy() && Val->getType()->isPointerTy()) {
        auto *PtrToInt = new llvm::PtrToIntInst(Val, g->target->is32Bit() ? LLVMTypes::Int32Type : LLVMTypes::Int64Type,
                                                "coal_diff_ty_ptr_cast", InsertBefore);
        return buildCast(PtrToInt, DstTy, InsertBefore);
    } else {
        return new llvm::BitCastInst(Val, DstTy, "coal_diff_ty_cast", InsertBefore);
    }
}

llvm::Value *MemoryCoalescing::buildEEI(llvm::Value *ExtractFrom, MemoryCoalescing::OffsetT OffsetBytes,
                                        llvm::Type *DstTy, llvm::Instruction *InsertBefore) const {
    Assert(!DstTy->isVectorTy() && !DstTy->isAggregateType() && "Expected scalar type");
    Assert(ExtractFrom->getType()->isVectorTy() && "Expected vector type");

    llvm::Type *ScalarType = ExtractFrom->getType()->getScalarType();
    unsigned ScalarTypeBytes = getScalarTypeSize(ScalarType);
    unsigned DstTyBytes = getScalarTypeSize(DstTy);
    unsigned Idx = OffsetBytes / ScalarTypeBytes;
    unsigned Rem = OffsetBytes % ScalarTypeBytes;
    llvm::Value *Res = nullptr;

    if (Rem != 0) {
        // Unaligned case: the resulting value starts inside Idx element.
        // TODO: this is handled via shuffle vector. Actually, it can be done
        // for all cases, but it is possible that such shuffle vector would
        // introduce redundant instructions. This should be investigated
        // at least on Xe target.

        // Cast source to byte vector
        Res = buildCast(
            ExtractFrom,
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
            llvm::FixedVectorType::get(LLVMTypes::Int8Type,
                                       ScalarTypeBytes *
                                           llvm::cast<llvm::FixedVectorType>(ExtractFrom->getType())->getNumElements()),
#else
            llvm::VectorType::get(LLVMTypes::Int8Type,
                                  ScalarTypeBytes *
                                      llvm::cast<llvm::VectorType>(ExtractFrom->getType())->getNumElements()),
#endif
            InsertBefore);
        // Prepare Idxs vector for shuffle vector
        std::vector<unsigned int> ByteIdxs(DstTyBytes);
        OffsetT CurrIdx = OffsetBytes;
        for (auto &Val : ByteIdxs)
            Val = CurrIdx++;
        llvm::ArrayRef<unsigned int> ByteIdxsArg(ByteIdxs);
        // Extract bytes via shuffle vector
        Res = new llvm::ShuffleVectorInst(Res, llvm::UndefValue::get(Res->getType()),
                                          llvm::ConstantDataVector::get(*g->ctx, ByteIdxsArg), "coal_unaligned_loader",
                                          InsertBefore);
        // Cast byte vector to scalar value
        Res = buildCast(Res, llvm::IntegerType::get(*g->ctx, /* NumBits */ DstTyBytes << 3), InsertBefore);
        // Cast to actual DstTy
        return buildCast(Res, DstTy, InsertBefore);
    }

    Res = llvm::ExtractElementInst::Create(ExtractFrom, llvm::ConstantInt::get(LLVMTypes::Int64Type, Idx),
                                           "mem_coal_eei", InsertBefore);
    if (DstTy == ScalarType) {
        // Done here
        return Res;
    } else if (DstTyBytes == ScalarTypeBytes) {
        // Just create bitcast
        return buildCast(Res, DstTy, InsertBefore);
    }

    // Smaller type case. Need to insert cast-eei chain.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    auto Cast = buildCast(Res, llvm::FixedVectorType::get(DstTy, ScalarTypeBytes / DstTyBytes), InsertBefore);
#else
    auto Cast = buildCast(Res, llvm::VectorType::get(DstTy, ScalarTypeBytes / DstTyBytes), InsertBefore);
#endif
    // Now call the builder with adjusted types
    return buildEEI(Cast, Rem, DstTy, InsertBefore);
}

llvm::Value *MemoryCoalescing::extractValueFromBlock(const MemoryCoalescing::BlockInstsVecT &BlockInstsVec,
                                                     MemoryCoalescing::OffsetT OffsetBytes, llvm::Type *DstTy,
                                                     llvm::Instruction *InsertBefore) const {
    Assert(BlockInstsVec.size() > 0);
    OffsetT BlockSizeInBytes = getScalarTypeSize(BlockInstsVec[0]->getType()) *
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
                               llvm::cast<llvm::FixedVectorType>(BlockInstsVec[0]->getType())->getNumElements();
#else
                               llvm::cast<llvm::VectorType>(BlockInstsVec[0]->getType())->getNumElements();
#endif
    unsigned StartIdx = OffsetBytes / BlockSizeInBytes;
    unsigned EndIdx = (OffsetBytes + getScalarTypeSize(DstTy) - 1) / BlockSizeInBytes;
    unsigned BlocksAffected = EndIdx - StartIdx + 1;
    Assert(EndIdx < BlockInstsVec.size());
    if (BlocksAffected == 1) {
        // Simple case: just get value needed
        return buildEEI(BlockInstsVec[StartIdx], OffsetBytes % BlockSizeInBytes, DstTy, InsertBefore);
    } else {
        // Need to get value from several blocks
        llvm::Value *ByteVec = llvm::UndefValue::get(
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
            llvm::FixedVectorType::get(LLVMTypes::Int8Type, getScalarTypeSize(DstTy)));
#else
            llvm::VectorType::get(LLVMTypes::Int8Type, getScalarTypeSize(DstTy)));
#endif
        for (OffsetT CurrOffset = OffsetBytes, TargetOffset = 0; CurrOffset < OffsetBytes + getScalarTypeSize(DstTy);
             ++CurrOffset, ++TargetOffset) {
            unsigned Idx = CurrOffset / BlockSizeInBytes;
            unsigned LocalOffset = CurrOffset % BlockSizeInBytes;
            llvm::Value *Elem = buildEEI(BlockInstsVec[Idx], LocalOffset, LLVMTypes::Int8Type, InsertBefore);
            ByteVec = buildIEI(ByteVec, Elem, TargetOffset, InsertBefore);
        }
        llvm::Value *ScalarizedByteVec = buildCast(
            ByteVec, llvm::IntegerType::get(*g->ctx, /* NumBits */ getScalarTypeSize(DstTy) << 3), InsertBefore);
        return buildCast(ScalarizedByteVec, DstTy, InsertBefore);
    }
}

class XeGatherCoalescing : public MemoryCoalescing {
  private:
    bool isOptimizationTarget(llvm::Instruction *Inst) const;
    llvm::Value *getPointer(llvm::Instruction *Inst) const;
    OffsetsVecT getOffset(llvm::Instruction *Inst) const;
    llvm::Value *getStoredValue(llvm::Instruction *Inst) const { return nullptr; }
    void optimizePtr(llvm::Value *Ptr, PtrData &PD, llvm::Instruction *InsertPoint);
    void runOnBasicBlock(llvm::BasicBlock &BB);

    llvm::CallInst *getPseudoGatherConstOffset(llvm::Instruction *Inst) const;

  public:
    static char ID;
    XeGatherCoalescing() : MemoryCoalescing(ID, MemoryCoalescing::MemType::OPT_LOAD, AddressSpace::ispc_global) {}

    llvm::StringRef getPassName() const { return "Xe Gather Coalescing"; }
};

char XeGatherCoalescing::ID = 0;

void XeGatherCoalescing::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("XeGatherCoalescing");
    runOnBasicBlockImpl(bb);
    DEBUG_END_PASS("XeGatherCoalescing");
}

void XeGatherCoalescing::optimizePtr(llvm::Value *Ptr, PtrData &PD, llvm::Instruction *InsertPoint) {
    // Analyse memory accesses
    OffsetT MinIdx = INT64_MAX;
    OffsetT MaxIdx = INT64_MIN;
    unsigned TotalMemOpsCounter = PD.InstsData.size();
    llvm::Type *LargestType = nullptr;
    unsigned LargestTypeSize = 0;
    for (auto &ID : PD.InstsData) {
        // Adjust borders
        for (auto Idx : ID.Offsets) {
            MaxIdx = std::max(Idx, MaxIdx);
            MinIdx = std::min(Idx, MinIdx);
        }
        // Largest type is needed to handle the case with different type sizes
        unsigned TypeSize = getScalarTypeSize(ID.Inst->getType());
        if (TypeSize > LargestTypeSize) {
            LargestTypeSize = TypeSize;
            LargestType = ID.Inst->getType()->getScalarType();
            if (LargestType->isPointerTy())
                LargestType = g->target->is32Bit() ? LLVMTypes::Int32Type : LLVMTypes::Int64Type;
        }
    }

    // Calculate data length
    Assert(LargestTypeSize > 0);
    uint64_t DataSize = MaxIdx - MinIdx + LargestTypeSize;
    // Calculate size of block loads in powers of two:
    // block loads are aligned to OWORDs
    unsigned ReqSize = 1;
    while (ReqSize < DataSize)
        ReqSize <<= 1;

    // Adjust req size and calculate num if insts needed
    // TODO: experiment showed performance improvement with
    // max ReqSize of 4 * OWORD instead of 8 * OWORD.
    // Further investigation is needed.
    unsigned MemInstsNeeded = 1;
    if (ReqSize > 4 * OWORD) {
        // Dealing with powers of two
        MemInstsNeeded = ReqSize / (4 * OWORD);
        ReqSize = 4 * OWORD;
    }

    // TODO: not clear if we should skip it
    if (ReqSize < OWORD) {
        // Skip for now.
        return;
    }

    // Check for threshold
    int MemOpsDiff = TotalMemOpsCounter - MemInstsNeeded;
    if (MemOpsDiff < g->opt.thresholdForXeGatherCoalescing) {
        return;
    }

    // Build block loads
    BlockInstsVecT BlockLDs;
    for (unsigned i = 0; i < MemInstsNeeded; ++i) {
        llvm::Constant *Offset = llvm::ConstantInt::get(LLVMTypes::Int64Type, MinIdx + i * ReqSize);
        llvm::PtrToIntInst *PtrToInt =
            new llvm::PtrToIntInst(Ptr, LLVMTypes::Int64Type, "vectorized_ptrtoint", InsertPoint);
        llvm::Instruction *Addr = llvm::BinaryOperator::CreateAdd(PtrToInt, Offset, "vectorized_address", InsertPoint);
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
        llvm::Type *RetType = llvm::FixedVectorType::get(LargestType, ReqSize / LargestTypeSize);
#else
        llvm::Type *RetType = llvm::VectorType::get(LargestType, ReqSize / LargestTypeSize);
#endif
        llvm::Instruction *LD = nullptr;
        if (g->opt.buildLLVMLoadsOnXeGatherCoalescing) {
            // Experiment: build standard llvm load instead of block ld
            llvm::IntToPtrInst *PtrForLd =
                new llvm::IntToPtrInst(Addr, llvm::PointerType::get(RetType, 0), "vectorized_address_ptr", InsertPoint);
            LD = new llvm::LoadInst(RetType, PtrForLd, "vectorized_ld_exp", InsertPoint);
        } else {
            llvm::Function *Fn = llvm::GenXIntrinsic::getGenXDeclaration(
                m->module, llvm::GenXIntrinsic::genx_svm_block_ld_unaligned, {RetType, Addr->getType()});
            LD = llvm::CallInst::Create(Fn, {Addr}, "vectorized_ld", InsertPoint);
        }
        BlockLDs.push_back(LD);
    }

    // Replace users
    for (auto &ID : PD.InstsData) {
        if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(ID.Inst)) {
            // Adjust offset
            OffsetT Offset = ID.Offsets[0] - MinIdx;
            LI->replaceAllUsesWith(extractValueFromBlock(BlockLDs, Offset, LI->getType(), InsertPoint));
        } else if (auto *Gather = getPseudoGatherConstOffset(ID.Inst)) {
            llvm::Value *NewVal = llvm::UndefValue::get(Gather->getType());
            unsigned CurrElem = 0;
            for (auto Offset : ID.Offsets) {
                // Adjust offset
                OffsetT AdjOffset = Offset - MinIdx;
                auto *ExtractedValue =
                    extractValueFromBlock(BlockLDs, AdjOffset, Gather->getType()->getScalarType(), InsertPoint);
                NewVal = llvm::InsertElementInst::Create(NewVal, ExtractedValue,
                                                         llvm::ConstantInt::get(LLVMTypes::Int64Type, CurrElem),
                                                         "gather_coal_iei", InsertPoint);
                ++CurrElem;
            }
            Gather->replaceAllUsesWith(NewVal);
        }

        // Mark to delete
        PossiblyDead.insert(ID.Inst);
    }

    // Done
    modifiedAny = true;
}

llvm::CallInst *XeGatherCoalescing::getPseudoGatherConstOffset(llvm::Instruction *Inst) const {
    if (auto CI = llvm::dyn_cast<llvm::CallInst>(Inst)) {
        llvm::Function *Function = CI->getCalledFunction();
        if (Function && Function->getName().startswith("__pseudo_gather_base_offsets"))
            if (llvm::isa<llvm::ConstantDataVector>(CI->getOperand(2)))
                return CI;
    }
    return nullptr;
}

bool XeGatherCoalescing::isOptimizationTarget(llvm::Instruction *Inst) const {
    if (auto LI = llvm::dyn_cast<llvm::LoadInst>(Inst)) {
        if (!LI->getType()->isVectorTy() && !LI->getType()->isAggregateType())
            return GetAddressSpace(LI->getPointerOperand()) == AddrSpace;
    } else if (auto Gather = getPseudoGatherConstOffset(Inst)) {
        return GetAddressSpace(getPointer(Gather)) == AddrSpace;
    }

    return false;
}

MemoryCoalescing::OffsetsVecT XeGatherCoalescing::getOffset(llvm::Instruction *Inst) const {
    OffsetsVecT Res;

    if (llvm::isa<llvm::LoadInst>(Inst))
        Res.push_back(0);
    else if (auto Gather = getPseudoGatherConstOffset(Inst)) {
        Res = getConstOffsetFromVector(Gather->getOperand(2));
        applyScale(Res, llvm::cast<llvm::ConstantInt>(Gather->getOperand(1))->getSExtValue());
    }

    return Res;
}

llvm::Value *XeGatherCoalescing::getPointer(llvm::Instruction *Inst) const {
    if (auto LI = llvm::dyn_cast<llvm::LoadInst>(Inst)) {
        return LI->getPointerOperand();
    } else if (auto Gather = getPseudoGatherConstOffset(Inst)) {
        return Gather->getOperand(0);
    }

    return nullptr;
}

static llvm::Pass *CreateXeGatherCoalescingPass() { return new XeGatherCoalescing; }

///////////////////////////////////////////////////////////////////////////
// ReplaceLLVMIntrinsics

/** This pass replaces LLVM intrinsics unsupported on Xe
 */

class ReplaceLLVMIntrinsics : public llvm::FunctionPass {
  public:
    static char ID;
    ReplaceLLVMIntrinsics() : FunctionPass(ID) {}
    llvm::StringRef getPassName() const { return "LLVM intrinsics replacement"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);
};

char ReplaceLLVMIntrinsics::ID = 0;

bool ReplaceLLVMIntrinsics::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("LLVM intrinsics replacement");
    std::vector<llvm::AllocaInst *> Allocas;

    bool modifiedAny = false;

restart:
    for (llvm::BasicBlock::iterator I = bb.begin(), E = --bb.end(); I != E; ++I) {
        llvm::Instruction *inst = &*I;
        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            llvm::Function *func = ci->getCalledFunction();
            if (func == NULL || !func->isIntrinsic())
                continue;

            if (func->getName().equals("llvm.trap")) {
                llvm::Type *argTypes[] = {LLVMTypes::Int1VectorType, LLVMTypes::Int16VectorType};
                // Description of parameters for genx_raw_send_noresult can be found in target-genx.ll
                auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(
                    m->module, llvm::GenXIntrinsic::genx_raw_send_noresult, argTypes);
                llvm::SmallVector<llvm::Value *, 8> Args;
                Args.push_back(llvm::ConstantInt::get(LLVMTypes::Int32Type, 0));
                Args.push_back(llvm::ConstantVector::getSplat(
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
                    g->target->getNativeVectorWidth(),
#elif ISPC_LLVM_VERSION < ISPC_LLVM_12_0
                    {static_cast<unsigned int>(g->target->getNativeVectorWidth()), false},
#else // LLVM 12.0+
                    llvm::ElementCount::get(static_cast<unsigned int>(g->target->getNativeVectorWidth()), false),
#endif
                    llvm::ConstantInt::getTrue(*g->ctx)));

                Args.push_back(llvm::ConstantInt::get(LLVMTypes::Int32Type, 39));
                Args.push_back(llvm::ConstantInt::get(LLVMTypes::Int32Type, 33554448));
                llvm::Value *zeroMask = llvm::ConstantVector::getSplat(
#if ISPC_LLVM_VERSION < ISPC_LLVM_11_0
                    g->target->getNativeVectorWidth(),
#elif ISPC_LLVM_VERSION < ISPC_LLVM_12_0
                    {static_cast<unsigned int>(g->target->getNativeVectorWidth()), false},
#else // LLVM 12.0+
                    llvm::ElementCount::get(static_cast<unsigned int>(g->target->getNativeVectorWidth()), false),
#endif
                    llvm::Constant::getNullValue(llvm::Type::getInt16Ty(*g->ctx)));
                Args.push_back(zeroMask);

                llvm::Instruction *newInst = llvm::CallInst::Create(Fn, Args, ci->getName());
                if (newInst != NULL) {
                    llvm::ReplaceInstWithInst(ci, newInst);
                    modifiedAny = true;
                    goto restart;
                }
            } else if (func->getName().equals("llvm.experimental.noalias.scope.decl")) {
                // These intrinsics are not supported by backend so remove them.
                ci->eraseFromParent();
                modifiedAny = true;
                goto restart;
            } else if (func->getName().contains("llvm.abs")) {
                // Replace llvm.asb with llvm.genx.aba.alternative
                Assert(ci->getOperand(0));
                llvm::Type *argType = ci->getOperand(0)->getType();

                llvm::Type *Tys[2];
                Tys[0] = func->getReturnType(); // return type
                Tys[1] = argType;               // value type

                llvm::GenXIntrinsic::ID xeAbsID =
                    argType->isIntOrIntVectorTy() ? llvm::GenXIntrinsic::genx_absi : llvm::GenXIntrinsic::genx_absf;
                auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(m->module, xeAbsID, Tys);
                Assert(Fn);
                llvm::Instruction *newInst = llvm::CallInst::Create(Fn, ci->getOperand(0), "");
                if (newInst != NULL) {
                    lCopyMetadata(newInst, ci);
                    llvm::ReplaceInstWithInst(ci, newInst);
                    modifiedAny = true;
                    goto restart;
                }
            }
        }
    }
    DEBUG_END_PASS("LLVM intrinsics replacement");
    return modifiedAny;
}

bool ReplaceLLVMIntrinsics::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("ReplaceLLVMIntrinsics::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateReplaceLLVMIntrinsics() { return new ReplaceLLVMIntrinsics(); }

///////////////////////////////////////////////////////////////////////////
// CheckUnsupportedInsts

/** This pass checks if there are any functions used which are not supported currently for Xe target,
    reports error and stops compilation.
 */

class CheckUnsupportedInsts : public llvm::FunctionPass {
  public:
    static char ID;
    CheckUnsupportedInsts(bool last = false) : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Check unsupported instructions for Xe target"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);
};

char CheckUnsupportedInsts::ID = 0;

bool CheckUnsupportedInsts::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("CheckUnsupportedInsts");
    bool modifiedAny = false;
    // This list contains regex expr for unsupported function names
    // To be extended

    for (llvm::BasicBlock::iterator I = bb.begin(), E = --bb.end(); I != E; ++I) {
        llvm::Instruction *inst = &*I;
        SourcePos pos;
        lGetSourcePosFromMetadata(inst, &pos);
        // Report error if double type is not supported by the target
        if (!g->target->hasFp64Support()) {
            for (int i = 0; i < (int)inst->getNumOperands(); ++i) {
                llvm::Type *t = inst->getOperand(i)->getType();
                if (t == LLVMTypes::DoubleType || t == LLVMTypes::DoublePointerType ||
                    t == LLVMTypes::DoubleVectorType || t == LLVMTypes::DoubleVectorPointerType) {
                    Error(pos, "\'double\' type is not supported by the target\n");
                }
            }
        }
    }
    DEBUG_END_PASS("CheckUnsupportedInsts");

    return modifiedAny;
}

bool CheckUnsupportedInsts::runOnFunction(llvm::Function &F) {
    llvm::TimeTraceScope FuncScope("CheckUnsupportedInsts::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateCheckUnsupportedInsts() { return new CheckUnsupportedInsts(); }

///////////////////////////////////////////////////////////////////////////
// MangleOpenCLBuiltins

/** This pass mangles SPIR-V OpenCL builtins used in Xe target file
 */

class MangleOpenCLBuiltins : public llvm::FunctionPass {
  public:
    static char ID;
    MangleOpenCLBuiltins(bool last = false) : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Mangle OpenCL builtins"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);
};

char MangleOpenCLBuiltins::ID = 0;

static std::string mangleMathOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName().startswith("__spirv_ocl") && "wrong argument: ocl builtin is expected");
    std::string mangledName;
    llvm::Type *retType = func.getReturnType();
    std::string funcName = func.getName().str();
    std::vector<llvm::Type *> ArgTy;
    // spirv OpenCL builtins are used for double types only
    Assert(retType->isVectorTy() && llvm::dyn_cast<llvm::FixedVectorType>(retType)->getElementType()->isDoubleTy() ||
           retType->isSingleValueType() && retType->isDoubleTy());
    if (retType->isVectorTy() && llvm::dyn_cast<llvm::FixedVectorType>(retType)->getElementType()->isDoubleTy()) {
        // Get vector width from retType. Required width may be different from target width
        // for example for 32-width targets
        ArgTy.push_back(llvm::FixedVectorType::get(LLVMTypes::DoubleType,
                                                   llvm::dyn_cast<llvm::FixedVectorType>(retType)->getNumElements()));
        // _DvWIDTH suffix is used in target file to differentiate scalar
        // and vector versions of intrinsics. Here we remove this
        // suffix and mangle the name.
        size_t pos = funcName.find("_DvWIDTH");
        if (pos != std::string::npos) {
            funcName.erase(pos, 8);
        }
    } else if (retType->isSingleValueType() && retType->isDoubleTy()) {
        ArgTy.push_back(LLVMTypes::DoubleType);
    }
    mangleOpenClBuiltin(funcName, ArgTy, mangledName);
    return mangledName;
}

static std::string manglePrintfOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName() == "__spirv_ocl_printf" && "wrong argument: ocl builtin is expected");
    std::string mangledName;
    mangleOpenClBuiltin(func.getName().str(), func.getArg(0)->getType(), mangledName);
    return mangledName;
}

static std::string mangleOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName().startswith("__spirv_ocl") && "wrong argument: ocl builtin is expected");
    if (func.getName() == "__spirv_ocl_printf")
        return manglePrintfOCLBuiltin(func);
    return mangleMathOCLBuiltin(func);
}

bool MangleOpenCLBuiltins::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("MangleOpenCLBuiltins");
    bool modifiedAny = false;
    for (llvm::BasicBlock::iterator I = bb.begin(), E = --bb.end(); I != E; ++I) {
        llvm::Instruction *inst = &*I;
        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            llvm::Function *func = ci->getCalledFunction();
            if (func == NULL)
                continue;
            if (func->getName().startswith("__spirv_ocl")) {
                std::string mangledName = mangleOCLBuiltin(*func);
                func->setName(mangledName);
                modifiedAny = true;
            }
        }
    }
    DEBUG_END_PASS("MangleOpenCLBuiltins");

    return modifiedAny;
}

bool MangleOpenCLBuiltins::runOnFunction(llvm::Function &F) {
    llvm::TimeTraceScope FuncScope("MangleOpenCLBuiltins::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateMangleOpenCLBuiltins() { return new MangleOpenCLBuiltins(); }

class DemotePHIs : public llvm::FunctionPass {
  public:
    static char ID;
    DemotePHIs() : FunctionPass(ID) {}
    llvm::StringRef getPassName() const { return "Demote PHI nodes"; }
    bool runOnFunction(llvm::Function &F);
};

char DemotePHIs::ID = 0;

bool DemotePHIs::runOnFunction(llvm::Function &F) {
    llvm::TimeTraceScope FuncScope("DemotePHIs::runOnFunction", F.getName());
    if (F.isDeclaration() || skipFunction(F))
        return false;
    std::vector<llvm::Instruction *> WorkList;
    for (auto &ibb : F)
        for (llvm::BasicBlock::iterator iib = ibb.begin(), iie = ibb.end(); iib != iie; ++iib)
            if (llvm::isa<llvm::PHINode>(iib))
                WorkList.push_back(&*iib);

    // Demote phi nodes
    for (auto *ilb : llvm::reverse(WorkList))
        DemotePHIToStack(llvm::cast<llvm::PHINode>(ilb), nullptr);

    return !WorkList.empty();
}

static llvm::Pass *CreateDemotePHIs() { return new DemotePHIs(); }

#endif
