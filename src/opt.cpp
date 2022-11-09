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
    @brief Implementation of ISPC optimization pipeline.
*/

#include "opt.h"
#include "ctx.h"
#include "llvmutil.h"
#include "module.h"
#include "opt/ISPCPass.h"
#include "opt/ISPCPasses.h"
#include "sym.h"
#include "util.h"

#include <map>
#include <regex>
#include <set>
#include <sstream>
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
#if ISPC_LLVM_VERSION >= ISPC_LLVM_15_0
#include <llvm/Transforms/Scalar/SimpleLoopUnswitch.h>
#endif
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Vectorize.h>

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
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include <llvm/GenXIntrinsics/GenXIntrOpts.h>
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#include <llvm/GenXIntrinsics/GenXSPIRVWriterAdaptor.h>
// Used for Xe gather coalescing
#include <llvm/Transforms/Utils/Local.h>

#endif

using namespace ispc;

#ifndef ISPC_NO_DUMPS
static llvm::Pass *CreateDebugPass(char *output);
static llvm::Pass *CreateDebugPassFile(int number, llvm::StringRef name, std::string dir);
#endif

static llvm::Pass *CreateReplaceStdlibShiftPass();

#ifdef ISPC_XE_ENABLED
static llvm::Pass *CreateXeGatherCoalescingPass();
static llvm::Pass *CreateReplaceLLVMIntrinsics();
static llvm::Pass *CreateDemotePHIs();
static llvm::Pass *CreateCheckIRForGenTarget();
static llvm::Pass *CreateMangleOpenCLBuiltins();
#endif

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
                PM.add(CreateDebugPassFile(number, P->getPassName(), g->dumpFilePath));
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
            // This pass is needed for correct prints work
            optPM.add(llvm::createSROAPass());
            optPM.add(CreateReplaceLLVMIntrinsics());
            optPM.add(CreateCheckIRForGenTarget());
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

#if ISPC_LLVM_VERSION < ISPC_LLVM_15_0
        // Starting LLVM 15.0 this pass is supported with new pass manager only (217e857)
        // TODO: switch ISPC to new pass manager: https://github.com/ispc/ispc/issues/2359
        optPM.add(llvm::createArgumentPromotionPass());
#endif

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
        }
#endif
        optPM.add(llvm::createTailCallEliminationPass());

        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.add(CreateIntrinsicsOptPass(), 250);
            optPM.add(CreateInstructionSimplifyPass());
        }

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.add(llvm::createInstructionCombiningPass(), 255);
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
#if ISPC_LLVM_VERSION < ISPC_LLVM_15_0
        // Starting LLVM 15.0 this pass is supported with new pass manager only (217e857)
        // TODO: switch ISPC to new pass manager: https://github.com/ispc/ispc/issues/2359
        optPM.add(llvm::createArgumentPromotionPass());
#endif

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
        // Loop unswitch pass was removed in LLVM 15.0 (fb4113).
        // Recommended replacement: createSimpleLoopUnswitchLegacyPass
#if ISPC_LLVM_VERSION < ISPC_LLVM_15_0
        optPM.add(llvm::createLoopUnswitchPass(false));
#else
        optPM.add(llvm::createSimpleLoopUnswitchLegacyPass(false));
#endif
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

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget() && g->opt.disableGatherScatterOptimizations == false &&
            g->target->getVectorWidth() > 1) {
            if (!g->opt.disableXeGatherCoalescing) {
                optPM.add(CreateXeGatherCoalescingPass(), 321);

                // Try the llvm provided load/store vectorizer
                optPM.add(llvm::createLoadStoreVectorizerPass(), 325);
            }
        }
#endif

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
            optPM.add(CreateCheckIRForGenTarget());
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
    DebugPassFile(int number, llvm::StringRef name, std::string dir)
        : ModulePass(ID), pnum(number), pname(name), pdir(dir) {}

    llvm::StringRef getPassName() const { return "Dump LLVM IR"; }
    bool runOnModule(llvm::Module &m);
    bool doInitialization(llvm::Module &m);

  private:
    void run(llvm::Module &m, bool init);
    int pnum;
    llvm::StringRef pname;
    std::string pdir;
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
    std::ostringstream oss;
    oss << (init ? "init_" : "ir_") << pnum << "_" << sanitize(std::string{pname}) << ".ll";

    const std::string pathFile{oss.str()};

#ifdef ISPC_HOST_IS_WINDOWS
    const std::string pathSep{"\\"};
#else
    const std::string pathSep{"/"};
#endif // ISPC_HOST_IS_WINDOWS

    std::string pathDirFile;

    if (!pdir.empty()) {
        llvm::sys::fs::create_directories(pdir);
        pathDirFile = pdir + pathSep + pathFile;
    } else {
        pathDirFile = pathFile;
    }

    std::error_code EC;
    llvm::raw_fd_ostream OS(pathDirFile, EC, llvm::sys::fs::OF_None);
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

static llvm::Pass *CreateDebugPassFile(int number, llvm::StringRef name, std::string dir) {
    return new DebugPassFile(number, name, dir);
}
#endif

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
        llvm::Type *tType = GEP->getSourceElementType();
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
    Assert(llvm::cast<llvm::GetElementPtrInst>(DanglingGEP));
    llvm::Type *tType = llvm::cast<llvm::GetElementPtrInst>(DanglingGEP)->getSourceElementType();

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
    bool isConstOffsetPseudoGather(llvm::CallInst *CI) const;

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
            if (isConstOffsetPseudoGather(CI))
                return CI;
    }
    return nullptr;
}

bool XeGatherCoalescing::isConstOffsetPseudoGather(llvm::CallInst *CI) const {
    Assert(CI != nullptr && CI->getCalledFunction() &&
           CI->getCalledFunction()->getName().startswith("__pseudo_gather_base_offsets"));
    llvm::Value *opOffset = CI->getOperand(2);

    return (llvm::isa<llvm::ConstantDataVector>(opOffset) || llvm::isa<llvm::ConstantAggregateZero>(opOffset) ||
            llvm::isa<llvm::ConstantVector>(opOffset));
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
        if (llvm::isa<llvm::ConstantAggregateZero>(Gather->getOperand(2))) {
            Res.push_back(0);
        } else {
            Res = getConstOffsetFromVector(Gather->getOperand(2));
            applyScale(Res, llvm::cast<llvm::ConstantInt>(Gather->getOperand(1))->getSExtValue());
        }
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
                    LLVMCopyMetadata(newInst, ci);
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
// CheckIRForGenTarget

/** This pass checks IR for Xe target and fix arguments for Xe intrinsics if needed.
    In case if unsupported statement is found, it reports error and stops compilation.
    Currently it performs 2 checks:
    1. double type support by target
    2. prefetch support by target and fixing prefetch args
 */

class CheckIRForGenTarget : public llvm::FunctionPass {
  public:
    static char ID;
    CheckIRForGenTarget(bool last = false) : FunctionPass(ID) {}

    llvm::StringRef getPassName() const { return "Check IR for Xe target"; }
    bool runOnBasicBlock(llvm::BasicBlock &BB);
    bool runOnFunction(llvm::Function &F);
};

char CheckIRForGenTarget::ID = 0;

bool CheckIRForGenTarget::runOnBasicBlock(llvm::BasicBlock &bb) {
    DEBUG_START_PASS("CheckIRForGenTarget");
    bool modifiedAny = false;
    // This list contains regex expr for unsupported function names
    // To be extended

    for (llvm::BasicBlock::iterator I = bb.begin(), E = --bb.end(); I != E; ++I) {
        llvm::Instruction *inst = &*I;
        SourcePos pos;
        LLVMGetSourcePosFromMetadata(inst, &pos);
        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            if (llvm::GenXIntrinsic::getGenXIntrinsicID(ci) == llvm::GenXIntrinsic::genx_lsc_prefetch_stateless) {
                // If prefetch is supported, fix data size parameter
                Assert(ci->arg_size() > 6);
                llvm::Value *dataSizeVal = ci->getArgOperand(6);
                llvm::ConstantInt *dataSizeConst = llvm::dyn_cast<llvm::ConstantInt>(dataSizeVal);
                Assert(dataSizeConst && (dataSizeConst->getBitWidth() == 8));
                int dataSizeNum = dataSizeConst->getSExtValue();
                // 0: invalid
                // 1: d8
                // 2: d16
                // 3: d32
                // 4: d64
                // Valid user's input is 1, 2, 4, 8
                int8_t genSize = 3;
                switch (dataSizeNum) {
                case 1:
                    genSize = 1;
                    break;
                case 2:
                    genSize = 2;
                    break;
                case 4:
                    genSize = 3;
                    break;
                case 8:
                    genSize = 4;
                    break;
                default:
                    Error(pos, "Incorrect data size argument for \'prefetch\'. Valid values are 1, 2, 4, 8");
                }
                llvm::Value *dataSizeGen = llvm::ConstantInt::get(LLVMTypes::Int8Type, genSize);
                ci->setArgOperand(6, dataSizeGen);
            }
        }
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
    DEBUG_END_PASS("CheckIRForGenTarget");

    return modifiedAny;
}

bool CheckIRForGenTarget::runOnFunction(llvm::Function &F) {
    llvm::TimeTraceScope FuncScope("CheckIRForGenTarget::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= runOnBasicBlock(BB);
    }
    return modifiedAny;
}

static llvm::Pass *CreateCheckIRForGenTarget() { return new CheckIRForGenTarget(); }

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
