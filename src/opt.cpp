/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file opt.cpp
    @brief Implementation of ISPC optimization pipeline.
*/

#include "opt.h"
#include "ctx.h"
#include "llvmutil.h"
#include "module.h"
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
#include <llvm/Analysis/OptimizationRemarkEmitter.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/PassRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO/ArgumentPromotion.h>
#include <llvm/Transforms/IPO/ConstantMerge.h>
#include <llvm/Transforms/IPO/DeadArgumentElimination.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/GlobalOpt.h>
#include <llvm/Transforms/IPO/Inliner.h>
#include <llvm/Transforms/IPO/SCCP.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Instrumentation.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Scalar/CorrelatedValuePropagation.h>
#include <llvm/Transforms/Scalar/DCE.h>
#include <llvm/Transforms/Scalar/DeadStoreElimination.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/IndVarSimplify.h>
#include <llvm/Transforms/Scalar/InstSimplifyPass.h>
#include <llvm/Transforms/Scalar/JumpThreading.h>
#include <llvm/Transforms/Scalar/LICM.h>
#include <llvm/Transforms/Scalar/LoopDeletion.h>
#include <llvm/Transforms/Scalar/LoopIdiomRecognize.h>
#include <llvm/Transforms/Scalar/LoopRotation.h>
#include <llvm/Transforms/Scalar/LoopUnrollPass.h>
#include <llvm/Transforms/Scalar/LowerExpectIntrinsic.h>
#include <llvm/Transforms/Scalar/LowerMatrixIntrinsics.h>
#include <llvm/Transforms/Scalar/MemCpyOptimizer.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SCCP.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimpleLoopUnswitch.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/TailRecursionElimination.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <llvm/Transforms/Vectorize/LoadStoreVectorizer.h>

#ifdef ISPC_HOST_IS_LINUX
#include <alloca.h>
#elif defined(ISPC_HOST_IS_WINDOWS)
#include <malloc.h>
#ifndef __MINGW32__
#define alloca _alloca
#endif
#endif // ISPC_HOST_IS_WINDOWS

#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXSPIRVWriterAdaptor.h>
#endif

using namespace ispc;

///////////////////////////////////////////////////////////////////////////
// This is a wrap over class llvm::ModulePassManager. This duplicates PassManager function run()
//   and adds several add functions with some checks and debug passes.
//   This wrap can control:
//   - If we want to switch off optimization with given number.
//   - If we want to dump LLVM IR after optimization with given number.
//   - If we want to generate LLVM IR debug for gdb after optimization with given number.
class DebugModulePassManager {
  public:
    DebugModulePassManager(llvm::Module &M, int optLevel) : m_number(0), m_optLevel(optLevel) {
        m = &M;
        llvm::Triple targetTriple = llvm::Triple(m->getTargetTriple());
        llvm::TargetLibraryInfoImpl *targetLibraryInfo = new llvm::TargetLibraryInfoImpl(targetTriple);
        llvm::TargetMachine *targetMachine = g->target->GetTargetMachine();

        // Create the new pass manager builder using our target machine.
        llvm::PassBuilder pb(targetMachine);

        // Register all the basic analyses with the managers.
        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        // Register all the analysis passes
        fam.registerPass([&] { return targetMachine->getTargetIRAnalysis(); });
        fam.registerPass([&] { return llvm::TargetLibraryAnalysis(*targetLibraryInfo); });

        // Add alias analysis for more aggressive optimizations
        if (m_optLevel != 0) {
            llvm::AAManager aam;
            aam.registerFunctionAnalysis<llvm::BasicAA>();
            aam.registerFunctionAnalysis<llvm::TypeBasedAA>();
            fam.registerPass([aam] { return std::move(aam); });
        }
    }

    template <typename T> void addModuleToLoopPass(T &&Pass, int stage = -1, bool memorySSA = false);
    template <typename T> void addModuleToPostOrderCGSCCPass(T &&Pass, int stage = -1);
    template <typename T> void addModuleToFunctionPass(T &&Pass, int stage = -1);
    template <typename T> void addModulePass(T &&Pass, int stage = -1);

    llvm::PreservedAnalyses run() { return mpm.run(*m, mam); }

  private:
    // Analysis managers
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::ModulePassManager mpm;
    llvm::Module *m;
    int m_number;
    int m_optLevel;

    template <typename T, typename F> void addPass(T &&Pass, const F &func, int stage = -1);
};

template <typename T> void DebugModulePassManager::addModuleToPostOrderCGSCCPass(T &&P, int stage) {
    addPass(
        std::move(P), [&](T &&P) { mpm.addPass(llvm::createModuleToPostOrderCGSCCPassAdaptor(std::move(P))); }, stage);
}

template <typename T> void DebugModulePassManager::addModuleToLoopPass(T &&P, int stage, bool memorySSA) {
    addPass(
        std::move(P),
        [&](T &&P) {
            mpm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::createFunctionToLoopPassAdaptor(
                std::move(P), memorySSA, /*UseBlockFrequencyInfo*/ memorySSA ? true : false)));
        },
        stage);
}

template <typename T> void DebugModulePassManager::addModuleToFunctionPass(T &&P, int stage) {
    addPass(
        std::move(P), [&](T &&P) { mpm.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(P))); }, stage);
}

template <typename T> void DebugModulePassManager::addModulePass(T &&P, int stage) {
    addPass(
        std::move(P), [&](T &&P) { mpm.addPass(std::move(P)); }, stage);
}

template <typename T, typename F> void DebugModulePassManager::addPass(T &&P, const F &func, int stage) {
    // taking number of optimization
    if (stage == -1) {
        m_number++;
    } else {
        m_number = stage;
    }
    auto className = T::name();
    if (g->off_stages.find(m_number) == g->off_stages.end()) {
        // adding optimization (not switched off)
        func(std::move(P));
        if (g->debug_stages.find(m_number) != g->debug_stages.end()) {
            // adding dump of LLVM IR after optimization
            if (g->dumpFile) {
                mpm.addPass(DebugPassFile(m_number, className, g->dumpFilePath));
            } else {
                char buf[100];
                snprintf(buf, sizeof(buf), "\n\n*****LLVM IR after phase %d: %s*****\n\n", m_number,
                         className.str().c_str());
                mpm.addPass(DebugPass(buf));
            }
        }
    }
}

void ispc::Optimize(llvm::Module *module, int optLevel) {
    if (g->debugPrint) {
        printf("*** Code going into optimization ***\n");
        module->print(llvm::errs(), nullptr);
    }
    DebugModulePassManager optPM(*module, optLevel);
    if (g->enableLLVMIntrinsics) {
        // Required for matrix intrinsics. This needs to happen before VerifierPass.
        // TODO : Limit pass to only when llvm.matrix.* intrinsics are used.
        optPM.addModuleToFunctionPass(llvm::LowerMatrixIntrinsicsPass()); // llvm.matrix
    }
    optPM.addModuleToFunctionPass(llvm::VerifierPass(), 0);

    optPM.addModuleToLoopPass(llvm::IndVarSimplifyPass());

    llvm::SimplifyCFGOptions simplifyCFGopt;
    simplifyCFGopt.HoistCommonInsts = true;
    if (optLevel == 0) {
        //  This is more or less the minimum set of optimizations that we
        //  need to do to generate code that will actually run.  (We can't
        //  run absolutely no optimizations, since the front-end needs us to
        //  take the various __pseudo_* functions it has emitted and turn
        //  them into something that can actually execute.
#ifdef ISPC_XE_ENABLED
        // mem2reg affects several acos/asin tests with O0 on Gen9,
        // seems like a problem with VC BE.
        if (g->target->isXeTarget()) {
            optPM.addModuleToFunctionPass(llvm::PromotePass());
        }
#endif
        optPM.addModuleToFunctionPass(ImproveMemoryOpsPass(), 100);

        if (g->opt.disableHandlePseudoMemoryOps == false)
            optPM.addModuleToFunctionPass(ReplacePseudoMemoryOpsPass());

        optPM.addModuleToFunctionPass(IntrinsicsOpt(), 102);
        optPM.addModuleToFunctionPass(IsCompileTimeConstantPass(true));
        optPM.addModulePass(llvm::ModuleInlinerWrapperPass());
        optPM.addModulePass(MakeInternalFuncsStaticPass());
        optPM.addModuleToFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addModulePass(llvm::GlobalDCEPass());

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.addModuleToFunctionPass(llvm::PromotePass());
            // This pass is needed for correct prints work
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
            optPM.addModuleToFunctionPass(llvm::SROAPass());
#else
            optPM.addModuleToFunctionPass(llvm::SROA());
#endif
            optPM.addModuleToFunctionPass(ReplaceLLVMIntrinsics());
            optPM.addModuleToFunctionPass(CheckIRForXeTarget());
            optPM.addModuleToFunctionPass(MangleOpenCLBuiltins());
            //  This pass is required to prepare LLVM IR for open source SPIR-V translator
            optPM.addModulePass(
                llvm::GenXSPIRVWriterAdaptor(true /*RewriteTypes*/, false /*RewriteSingleElementVectors*/));
            optPM.addModulePass(llvm::GlobalDCEPass());
        }
#endif
    } else {
        optPM.addModulePass(llvm::GlobalDCEPass(), 184);

        // Setup to use LLVM default AliasAnalysis
        // Ideally, we want call:
        //    llvm::PassManagerBuilder pm_Builder;
        //    pm_Builder.OptLevel = optLevel;
        //    pm_Builder.addInitialAliasAnalysisPasses(optPM);
        // but the addInitialAliasAnalysisPasses() is a private function
        // so we explicitly enable them here.
        // Need to keep sync with future LLVM change
        // An alternative is to call populateFunctionPassManager()

        optPM.addModuleToFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt), 192);
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        optPM.addModuleToFunctionPass(llvm::SROAPass());
#else
        optPM.addModuleToFunctionPass(llvm::SROA());
#endif
        optPM.addModuleToFunctionPass(llvm::EarlyCSEPass());
        optPM.addModuleToFunctionPass(llvm::LowerExpectIntrinsicPass());

        // Early optimizations to try to reduce the total amount of code to
        // work with if we can
        optPM.addModuleToFunctionPass(llvm::ReassociatePass(), 200);
        optPM.addModuleToFunctionPass(llvm::InstSimplifyPass());
        optPM.addModuleToFunctionPass(llvm::DCEPass());
        optPM.addModuleToFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addModuleToFunctionPass(llvm::PromotePass());
        optPM.addModuleToFunctionPass(llvm::ADCEPass());

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.addModuleToFunctionPass(llvm::InstCombinePass(), 210);
            optPM.addModuleToFunctionPass(ImproveMemoryOpsPass());
        }
        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.addModuleToFunctionPass(IntrinsicsOpt(), 215);
            optPM.addModuleToFunctionPass(InstructionSimplifyPass());
        }
        optPM.addModuleToFunctionPass(llvm::DCEPass(), 220);

        // On to more serious optimizations
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        optPM.addModuleToFunctionPass(llvm::SROAPass());
#else
        optPM.addModuleToFunctionPass(llvm::SROA());
#endif
        optPM.addModuleToFunctionPass(llvm::InstCombinePass());
        optPM.addModuleToFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addModuleToFunctionPass(llvm::PromotePass());
        optPM.addModulePass(llvm::GlobalOptPass());
        optPM.addModuleToFunctionPass(llvm::ReassociatePass());
        optPM.addModulePass(llvm::IPSCCPPass());
        optPM.addModuleToFunctionPass(ReplaceStdlibShiftPass(), 229);
        optPM.addModulePass(llvm::DeadArgumentEliminationPass(), 230);
        optPM.addModuleToFunctionPass(llvm::InstCombinePass());
        optPM.addModuleToFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));

        //  No such pass with new PM
        //  https://reviews.llvm.org/D44415
        //  optPM.add(llvm::createPruneEHPass());
        optPM.addModuleToPostOrderCGSCCPass(llvm::PostOrderFunctionAttrsPass());
        optPM.addModulePass(llvm::ReversePostOrderFunctionAttrsPass());

        // Next inline pass will remove functions, saved by __keep_funcs_live
        optPM.addModulePass(llvm::ModuleInlinerWrapperPass());
        optPM.addModuleToFunctionPass(llvm::InstSimplifyPass());
        optPM.addModuleToFunctionPass(llvm::DCEPass());
        optPM.addModuleToFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addModuleToPostOrderCGSCCPass(llvm::ArgumentPromotionPass());
        optPM.addModuleToFunctionPass(llvm::ADCEPass());
        optPM.addModuleToFunctionPass(llvm::InstCombinePass(), 241);
        optPM.addModuleToFunctionPass(llvm::JumpThreadingPass());
        optPM.addModuleToFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        optPM.addModuleToFunctionPass(llvm::SROAPass());
#else
        optPM.addModuleToFunctionPass(llvm::SROA());
#endif
        optPM.addModuleToFunctionPass(llvm::InstCombinePass());

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            // Inline
            optPM.addModuleToFunctionPass(llvm::CorrelatedValuePropagationPass());
            optPM.addModuleToFunctionPass(llvm::InstCombinePass());
            optPM.addModulePass(llvm::GlobalDCEPass());
            optPM.addModuleToFunctionPass(llvm::InstCombinePass());
            optPM.addModuleToFunctionPass(llvm::EarlyCSEPass());
            optPM.addModulePass(llvm::GlobalDCEPass());
        }
#endif
        optPM.addModuleToFunctionPass(llvm::TailCallElimPass());

        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.addModuleToFunctionPass(IntrinsicsOpt(), 250);
            optPM.addModuleToFunctionPass(InstructionSimplifyPass());
        }

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.addModuleToFunctionPass(llvm::InstCombinePass(), 255);
            optPM.addModuleToFunctionPass(ImproveMemoryOpsPass());

            if (g->opt.disableCoalescing == false) {
                // It is important to run this here to make it easier to
                // finding matching gathers we can coalesce..
                optPM.addModuleToFunctionPass(llvm::EarlyCSEPass(), 260);
                optPM.addModuleToFunctionPass(GatherCoalescePass());
            }
        }

        optPM.addModulePass(llvm::ModuleInlinerWrapperPass(), 265);
        optPM.addModuleToFunctionPass(llvm::InstSimplifyPass());
        optPM.addModuleToFunctionPass(IntrinsicsOpt());
        optPM.addModuleToFunctionPass(InstructionSimplifyPass());

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.addModuleToFunctionPass(llvm::InstCombinePass(), 270);
            optPM.addModuleToFunctionPass(ImproveMemoryOpsPass());
        }

        optPM.addModulePass(llvm::IPSCCPPass(), 275);
        optPM.addModulePass(llvm::DeadArgumentEliminationPass());
        optPM.addModuleToFunctionPass(llvm::ADCEPass());
        optPM.addModuleToFunctionPass(llvm::InstCombinePass());
        optPM.addModuleToFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));

        if (g->opt.disableHandlePseudoMemoryOps == false) {
            optPM.addModuleToFunctionPass(ReplacePseudoMemoryOpsPass(), 280);
        }
        optPM.addModuleToFunctionPass(IntrinsicsOpt(), 281);
        optPM.addModuleToFunctionPass(InstructionSimplifyPass());

        optPM.addModulePass(llvm::ModuleInlinerWrapperPass());
        optPM.addModuleToPostOrderCGSCCPass(llvm::ArgumentPromotionPass());

#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        optPM.addModuleToFunctionPass(llvm::SROAPass());
#else
        optPM.addModuleToFunctionPass(llvm::SROA());
#endif

        optPM.addModuleToFunctionPass(llvm::InstCombinePass());
        optPM.addModuleToFunctionPass(InstructionSimplifyPass());
        optPM.addModuleToFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addModuleToFunctionPass(llvm::ReassociatePass());

        // We provide the opt remark emitter pass for LICM to use.
        optPM.addModuleToFunctionPass(
            llvm::RequireAnalysisPass<llvm::OptimizationRemarkEmitterAnalysis, llvm::Function>());
        // Loop passes using MemorySSA
        optPM.addModuleToLoopPass(llvm::LoopRotatePass(), 290, true);
#if ISPC_LLVM_VERSION >= ISPC_LLVM_15_0
        // Use LLVM default options
        llvm::LICMOptions licmOpts;
        optPM.addModuleToLoopPass(llvm::LICMPass(licmOpts), 291, true);
#else
        optPM.addModuleToLoopPass(llvm::LICMPass(), 291, true);
#endif
        if (!g->target->isXeTarget()) {
            // SimpleLoopUnswitch is not a full equivalent of LoopUnswitch pass.
            // It produces much more basic blocks than LoopUnswitch which is
            // not efficient for Xe targets. Moreover when this pass is used
            // some integer division tests are failing on TGLLP Windows.
            // Disable this pass on Xe until the problem is fixed on BE side.
            optPM.addModuleToLoopPass(llvm::SimpleLoopUnswitchPass(false), 292, true);
        }

        optPM.addModuleToFunctionPass(llvm::InstCombinePass());
        optPM.addModuleToFunctionPass(InstructionSimplifyPass());

        optPM.addModuleToLoopPass(llvm::IndVarSimplifyPass());
        // Currently VC BE does not support memset/memcpy
        // so this pass is temporary disabled for Xe.
        if (!g->target->isXeTarget()) {
            optPM.addModuleToLoopPass(llvm::LoopIdiomRecognizePass());
        }

        optPM.addModuleToLoopPass(llvm::LoopDeletionPass());
        if (g->opt.unrollLoops) {
            optPM.addModuleToFunctionPass(llvm::LoopUnrollPass(), 300);
        }

        // Still use old GVN pass for 1:1 mapping with new PM
        // optPM.addModuleToFunctionPass(llvm::NewGVNPass(), 301);
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        optPM.addModuleToFunctionPass(llvm::GVNPass(), 301);
#else
        optPM.addModuleToFunctionPass(llvm::GVN(), 301);
#endif
        optPM.addModuleToFunctionPass(IsCompileTimeConstantPass(true));
        optPM.addModuleToFunctionPass(IntrinsicsOpt());
        optPM.addModuleToFunctionPass(InstructionSimplifyPass());

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget() && g->opt.disableGatherScatterOptimizations == false &&
            g->target->getVectorWidth() > 1) {
            if (!g->opt.disableXeGatherCoalescing) {
                optPM.addModuleToFunctionPass(XeGatherCoalescing(), 321);

                // Try the llvm provided load/store vectorizer
                optPM.addModuleToFunctionPass(llvm::LoadStoreVectorizerPass(), 325);
            }
        }
#endif
        // Currently VC BE does not support memset/memcpy
        // so this pass is temporary disabled for Xe.
        if (!g->target->isXeTarget()) {
            optPM.addModuleToFunctionPass(llvm::MemCpyOptPass());
        }
        optPM.addModuleToFunctionPass(llvm::SCCPPass());
        optPM.addModuleToFunctionPass(llvm::InstCombinePass());
        optPM.addModuleToFunctionPass(InstructionSimplifyPass());
        optPM.addModuleToFunctionPass(llvm::JumpThreadingPass());
        optPM.addModuleToFunctionPass(llvm::CorrelatedValuePropagationPass());
        optPM.addModuleToFunctionPass(llvm::DSEPass());
        optPM.addModuleToFunctionPass(llvm::ADCEPass());
        optPM.addModuleToFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addModuleToFunctionPass(llvm::InstCombinePass());
        optPM.addModuleToFunctionPass(InstructionSimplifyPass());
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.addModuleToFunctionPass(ReplaceLLVMIntrinsics());
        }
#endif

        optPM.addModuleToFunctionPass(PeepholePass());
        optPM.addModulePass(llvm::ModuleInlinerWrapperPass());
        optPM.addModuleToFunctionPass(llvm::ADCEPass());
        optPM.addModulePass(llvm::StripDeadPrototypesPass());
        optPM.addModulePass(MakeInternalFuncsStaticPass());
        optPM.addModulePass(llvm::GlobalDCEPass());
        optPM.addModulePass(llvm::ConstantMergePass());
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.addModuleToFunctionPass(CheckIRForXeTarget());
            optPM.addModuleToFunctionPass(MangleOpenCLBuiltins());
            //  This pass is required to prepare LLVM IR for open source SPIR-V translator
            optPM.addModulePass(
                llvm::GenXSPIRVWriterAdaptor(true /*RewriteTypes*/, false /*RewriteSingleElementVectors*/));
        }
#endif
    }

    // Finish up by making sure we didn't mess anything up in the IR along
    // the way.
    optPM.addModulePass(llvm::VerifierPass(), LAST_OPT_NUMBER);
    optPM.run();

    if (g->debugPrint) {
        printf("\n*****\nFINAL OUTPUT\n*****\n");
        module->print(llvm::errs(), nullptr);
    }
}
