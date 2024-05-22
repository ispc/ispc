/*
  Copyright (c) 2010-2024, Intel Corporation

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
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/Analysis/GlobalsModRef.h>
#include <llvm/Analysis/OptimizationRemarkEmitter.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/Analysis/ScopedNoAliasAA.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/DataLayout.h>
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
#include <llvm/IRPrinter/IRPrintingPasses.h>
#else
#include <llvm/IR/IRPrintingPasses.h>
#endif
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/PassRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#if ISPC_LLVM_VERSION >= ISPC_LLVM_17_0
#include <llvm/TargetParser/Triple.h>
#else
#include <llvm/ADT/Triple.h>
#endif
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
#if ISPC_LLVM_VERSION >= ISPC_LLVM_18_1
#include <llvm/Transforms/Scalar/InferAlignment.h>
#endif
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
#include <llvm/Transforms/Scalar/NewGVN.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SCCP.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimpleLoopUnswitch.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/TailRecursionElimination.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <llvm/Transforms/Vectorize/LoadStoreVectorizer.h>

#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXSPIRVWriterAdaptor.h>
#endif

using namespace ispc;
// Strips all non-alphanumeric characters from given string.
static std::string lSanitize(std::string in) {
    llvm::Regex r("[^[:alnum:]]");
    while (r.match(in))
        in = r.sub("", in);
    return in;
}

// Get path to dump file
static std::string getDumpFilePath(std::string className, int pnum) {
    std::ostringstream oss;
    oss << "ir_" << pnum << "_" << lSanitize(std::string{className}) << ".ll";

    const std::string pathFile{oss.str()};

#ifdef ISPC_HOST_IS_WINDOWS
    const std::string pathSep{"\\"};
#else
    const std::string pathSep{"/"};
#endif // ISPC_HOST_IS_WINDOWS

    std::string pathDirFile;
    if (!g->dumpFilePath.empty()) {
        llvm::sys::fs::create_directories(g->dumpFilePath);
        pathDirFile = g->dumpFilePath + pathSep + pathFile;
    } else {
        pathDirFile = pathFile;
    }
    return pathDirFile;
}

DebugModulePassManager::DebugModulePassManager(llvm::Module &M, int optLevel) : m_passNumber(0), m_optLevel(optLevel) {
    m = &M;
    llvm::Triple targetTriple = llvm::Triple(m->getTargetTriple());
    llvm::TargetLibraryInfoImpl targetLibraryInfo(targetTriple);
    targetMachine = g->target->GetTargetMachine();

    // We have to register an llvm::OptNoneInstrumentation with a llvm::PassInstrumentationCallbacks,
    // which is then registered in the llvm::PassBuilder constructor.
    // This ensures that any function with optnone will not be optimized.
    OptNoneInst.registerCallbacks(PIC);

    if (g->debugPMTimeTrace) {
        // Enable time traces for optimization passes.
        TimePasses.registerCallbacks(PIC);
    }
    // Create the new pass manager builder using our target machine.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
    pb = llvm::PassBuilder(targetMachine, llvm::PipelineTuningOptions(), std::nullopt, &PIC);
#else
    pb = llvm::PassBuilder(targetMachine, llvm::PipelineTuningOptions(), llvm::None, &PIC);
#endif

    // Register all the basic analyses with the managers.
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

#if ISPC_LLVM_VERSION >= ISPC_LLVM_17_0
    SI.registerCallbacks(PIC, &mam);
#else
    SI.registerCallbacks(PIC, &fam);
#endif

    // Register all the analysis passes
    fam.registerPass([&] { return targetMachine->getTargetIRAnalysis(); });
    fam.registerPass([&] { return llvm::TargetLibraryAnalysis(targetLibraryInfo); });

    // Add alias analysis for more aggressive optimizations
    if (m_optLevel != 0) {
        llvm::AAManager aam;
        // The order in which these are registered determines their priority when
        // being queried.

        // First we register the basic alias analysis that provides the majority of
        // per-function local AA logic. This is a stateless, on-demand local set of
        // AA techniques.
        aam.registerFunctionAnalysis<llvm::BasicAA>();

        // Next we query fast, specialized alias analyses that wrap IR-embedded
        // information about aliasing.
        aam.registerFunctionAnalysis<llvm::ScopedNoAliasAA>();
        aam.registerFunctionAnalysis<llvm::TypeBasedAA>();

        // Add support for querying global aliasing information when available.
        // Because the `AAManager` is a function analysis and `GlobalsAA` is a module
        // analysis, all that the `AAManager` can do is query for any *cached*
        // results from `GlobalsAA` through a readonly proxy.
        // aam.registerModuleAnalysis<llvm::GlobalsAA>();

        // Add target-specific alias analyses.
        if (targetMachine) {
            targetMachine->registerDefaultAliasAnalyses(aam);
        }
        fam.registerPass([aam] { return std::move(aam); });
    }
}

llvm::PreservedAnalyses DebugModulePassManager::run() { return mpm.run(*m, mam); }

void DebugModulePassManager::setMemorySSA(bool v) { m_memorySSA = v; }

void DebugModulePassManager::setBlocksFreq(bool v) { m_blocksFreq = v; }

// Add pass to pass manager and print IR if needed
void DebugModulePassManager::addPassAndDebugPrint(std::string name, DebugModulePassManager::Passes kind) {
    if (g->off_stages.find(m_passNumber) == g->off_stages.end()) {
        if (g->debug_stages.find(m_passNumber) != g->debug_stages.end()) {
            char banner[100];
            snprintf(banner, sizeof(banner), "\n\n; *****LLVM IR after phase : %s*****\n\n", name.c_str());
            llvm::raw_ostream *outputStream = nullptr;
            if (g->dumpFile) {
                std::error_code EC;
                std::unique_ptr<llvm::raw_fd_ostream> outFile = std::make_unique<llvm::raw_fd_ostream>(
                    getDumpFilePath(name, m_passNumber), EC, llvm::sys::fs::OF_None);
                if (!EC) {
                    outputDebugDumps.push_back(std::move(outFile));
                    outputStream = outputDebugDumps.back().get();
                }
            }
            if (g->dumpFile) {
                if (kind == Passes::Function) {
                    commitFunctionToModulePassManager();
                    mpm.addPass(llvm::PrintModulePass(outputStream ? *outputStream : llvm::outs(), banner));
                    initFunctionPassManager();
                } else if (kind == Passes::Loop) {
                    commitLoopToFunctionPassManager();
                    commitFunctionToModulePassManager();
                    mpm.addPass(llvm::PrintModulePass(outputStream ? *outputStream : llvm::outs(), banner));
                    initFunctionPassManager();
                    initLoopPassManager();
                } else if (kind == Passes::Module) {
                    mpm.addPass(llvm::PrintModulePass(outputStream ? *outputStream : llvm::outs(), banner));
                }
            } else {
                if (kind == Passes::Function) {
                    fpmVec.back()->addPass(
                        llvm::PrintFunctionPass(outputStream ? *outputStream : llvm::outs(), banner));
                } else if (kind == Passes::Module) {
                    mpm.addPass(llvm::PrintModulePass(outputStream ? *outputStream : llvm::outs(), banner));
                } else if (kind == Passes::Loop) {
                    lpmVec.back()->addPass(llvm::PrintLoopPass(outputStream ? *outputStream : llvm::outs(), banner));
                }
            }
        }
    }
}

// Start a new group of function passes
void DebugModulePassManager::initFunctionPassManager() {
    Assert(!m_isFPMOpen && "FunctionPassManager has been already initialized");
    auto fpm = std::make_unique<llvm::FunctionPassManager>();
    fpmVec.push_back(std::move(fpm));
    m_isFPMOpen = true;
}

// Add function passes to the ModulePassManager
void DebugModulePassManager::commitFunctionToModulePassManager() {
    Assert(m_isFPMOpen && "FunctionPassManager has not been initialized or already committed.");
    if (fpmVec.empty()) {
        return;
    }
    // Get the last element of fpmVec
    llvm::FunctionPassManager *lastFPM = fpmVec.back().get();
    mpm.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(*lastFPM)));
    m_isFPMOpen = false;
}

// Start a new group of loop passes
void DebugModulePassManager::initLoopPassManager() {
    Assert(!m_isLPMOpen && "LoopPassManager has been already initialized");
    auto lpm = std::make_unique<llvm::LoopPassManager>();
    lpmVec.push_back(std::move(lpm));
    m_isLPMOpen = true;
}

// Add loop passes to the FunctionPassManager
void DebugModulePassManager::commitLoopToFunctionPassManager() {
    Assert(m_isLPMOpen && "LoopPassManager has not been initialized or already committed.");
    if (fpmVec.empty() || lpmVec.empty()) {
        return;
    }
    // Get the last element of lpmVec
    llvm::LoopPassManager *lastLPM = lpmVec.back().get();
    fpmVec.back()->addPass(llvm::createFunctionToLoopPassAdaptor(std::move(*lastLPM), m_memorySSA, m_blocksFreq));
    m_isLPMOpen = false;
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
        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::LowerMatrixIntrinsicsPass()); // llvm.matrix
        optPM.commitFunctionToModulePassManager();
    }
    optPM.addModulePass(llvm::VerifierPass(), 0);

    optPM.initFunctionPassManager();
    optPM.initLoopPassManager();
    optPM.addLoopPass(llvm::IndVarSimplifyPass());
    optPM.commitLoopToFunctionPassManager();
    optPM.commitFunctionToModulePassManager();

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
            optPM.initFunctionPassManager();
            optPM.addFunctionPass(llvm::PromotePass());
            optPM.commitFunctionToModulePassManager();
        }
#endif
        optPM.initFunctionPassManager();
        optPM.addFunctionPass(ImproveMemoryOpsPass(), 100);

        if (g->opt.disableHandlePseudoMemoryOps == false)
            optPM.addFunctionPass(ReplacePseudoMemoryOpsPass());

        optPM.addFunctionPass(IntrinsicsOpt(), 102);
        optPM.addFunctionPass(IsCompileTimeConstantPass(true));
        optPM.commitFunctionToModulePassManager();

        optPM.addModulePass(llvm::ModuleInlinerWrapperPass());
        optPM.addModulePass(MakeInternalFuncsStaticPass());

        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.commitFunctionToModulePassManager();

        optPM.addModulePass(llvm::GlobalDCEPass());

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.initFunctionPassManager();
            optPM.addFunctionPass(llvm::PromotePass());
            // This pass is needed for correct prints work
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
            // We don't have any LICM or SimplifyCFG passes scheduled after us, that would cleanup
            // the CFG mess SROAPass may created if allowed to modify CFG, so forbid that.
            optPM.addFunctionPass(llvm::SROAPass(llvm::SROAOptions::PreserveCFG));
#elif ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
            optPM.addFunctionPass(llvm::SROAPass());
#else
            optPM.addFunctionPass(llvm::SROA());
#endif
            optPM.addFunctionPass(ReplaceLLVMIntrinsics());
            optPM.addFunctionPass(CheckIRForXeTarget());
            optPM.addFunctionPass(MangleOpenCLBuiltins());
            optPM.commitFunctionToModulePassManager();
            //  This pass is required to prepare LLVM IR for open source SPIR-V translator
            optPM.addModulePass(
                llvm::GenXSPIRVWriterAdaptor(true /*RewriteTypes*/, false /*RewriteSingleElementVectors*/));
            optPM.addModulePass(llvm::GlobalDCEPass());
        }
#endif
    } else {
        optPM.addModulePass(llvm::GlobalDCEPass(), 184);

        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt), 192);
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
        optPM.addFunctionPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
#elif ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        optPM.addFunctionPass(llvm::SROAPass());
#else
        optPM.addFunctionPass(llvm::SROA());
#endif
        optPM.addFunctionPass(llvm::EarlyCSEPass());
        optPM.addFunctionPass(llvm::LowerExpectIntrinsicPass());

        // Early optimizations to try to reduce the total amount of code to
        // work with if we can
        optPM.addFunctionPass(llvm::ReassociatePass(), 200);
        optPM.addFunctionPass(llvm::InstSimplifyPass());
        optPM.addFunctionPass(llvm::DCEPass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addFunctionPass(llvm::PromotePass());
        optPM.addFunctionPass(llvm::ADCEPass());

#if ISPC_LLVM_VERSION >= ISPC_LLVM_18_1
        // Note: this pass has been added since LLVM 18.1.
        // InstCombine contains similar functionality. It can be enabled back
        // (at the moment) with enable-infer-alignment-pass=false option.
        // It looks like it is enough to call it once before the last call to
        // InstCombine pass. But maybe more clear way to preserve previous
        // functionality is to call it before InstCombine every time.
        // Let us call it once before the first InstCombine invocation and
        // before the last one.
        optPM.addFunctionPass(llvm::InferAlignmentPass());
#endif
        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.addFunctionPass(llvm::InstCombinePass(), 210);
            optPM.addFunctionPass(ImproveMemoryOpsPass());
        }
        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.addFunctionPass(IntrinsicsOpt(), 215);
            optPM.addFunctionPass(InstructionSimplifyPass());
        }
        optPM.addFunctionPass(llvm::DCEPass(), 220);

        // On to more serious optimizations
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
        optPM.addFunctionPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
#elif ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        optPM.addFunctionPass(llvm::SROAPass());
#else
        optPM.addFunctionPass(llvm::SROA());
#endif
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addFunctionPass(llvm::PromotePass());
        optPM.addFunctionPass(llvm::ReassociatePass());
        optPM.setBlocksFreq(true);
        optPM.initLoopPassManager();
        optPM.addLoopPass(llvm::LoopFullUnrollPass());
        optPM.commitLoopToFunctionPassManager();
        optPM.setBlocksFreq(false);
        optPM.addFunctionPass(ReplaceStdlibShiftPass(), 229);
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.commitFunctionToModulePassManager();

        optPM.addModulePass(llvm::GlobalOptPass());
        optPM.addModulePass(llvm::IPSCCPPass());
        optPM.addModulePass(llvm::DeadArgumentEliminationPass());

        //  No such pass with new PM
        //  https://reviews.llvm.org/D44415
        //  optPM.add(llvm::createPruneEHPass());
        optPM.addPostOrderCGSCCPass(llvm::PostOrderFunctionAttrsPass());
        optPM.addModulePass(llvm::ReversePostOrderFunctionAttrsPass());

        // Next inline pass will remove functions, saved by __keep_funcs_live
        optPM.addModulePass(llvm::ModuleInlinerWrapperPass());

        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::InstSimplifyPass());
        optPM.addFunctionPass(llvm::DCEPass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addFunctionPass(llvm::ADCEPass());
        optPM.addFunctionPass(llvm::InstCombinePass(), 241);
        optPM.addFunctionPass(llvm::JumpThreadingPass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
        optPM.addFunctionPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
#elif ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        optPM.addFunctionPass(llvm::SROAPass());
#else
        optPM.addFunctionPass(llvm::SROA());
#endif
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.commitFunctionToModulePassManager();

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            // Inline
            optPM.initFunctionPassManager();
            optPM.addFunctionPass(llvm::CorrelatedValuePropagationPass());
            optPM.addFunctionPass(llvm::InstCombinePass());
            optPM.commitFunctionToModulePassManager();
            optPM.addModulePass(llvm::GlobalDCEPass());
            optPM.initFunctionPassManager();
            optPM.addFunctionPass(llvm::InstCombinePass());
            optPM.addFunctionPass(llvm::EarlyCSEPass());
            optPM.commitFunctionToModulePassManager();
            optPM.addModulePass(llvm::GlobalDCEPass());
        }
#endif
        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::TailCallElimPass());

        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.addFunctionPass(IntrinsicsOpt(), 250);
            optPM.addFunctionPass(InstructionSimplifyPass());
        }

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.addFunctionPass(llvm::InstCombinePass(), 255);
            optPM.addFunctionPass(ImproveMemoryOpsPass());

            if (g->opt.disableCoalescing == false) {
                // It is important to run this here to make it easier to
                // finding matching gathers we can coalesce..
                optPM.addFunctionPass(llvm::EarlyCSEPass(), 260);
                optPM.addFunctionPass(GatherCoalescePass());
            }
        }
        optPM.commitFunctionToModulePassManager();
        optPM.addModulePass(llvm::ModuleInlinerWrapperPass(), 265);
        // If we didn't decide to inline a function, check to see if we can
        // transform it to pass arguments by value instead of by reference.
        optPM.addPostOrderCGSCCPass(llvm::ArgumentPromotionPass());

        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::InstSimplifyPass());
        optPM.addFunctionPass(IntrinsicsOpt());
        optPM.addFunctionPass(InstructionSimplifyPass());

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.addFunctionPass(llvm::InstCombinePass(), 270);
            optPM.addFunctionPass(ImproveMemoryOpsPass());
        }
        optPM.commitFunctionToModulePassManager();
        optPM.addModulePass(llvm::IPSCCPPass(), 275);
        optPM.addModulePass(llvm::DeadArgumentEliminationPass());

        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::ADCEPass());
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));

        if (g->opt.disableHandlePseudoMemoryOps == false) {
            optPM.addFunctionPass(ReplacePseudoMemoryOpsPass(), 280);
        }
        optPM.addFunctionPass(IntrinsicsOpt(), 281);
        optPM.addFunctionPass(InstructionSimplifyPass());
        optPM.commitFunctionToModulePassManager();

        optPM.addModulePass(llvm::ModuleInlinerWrapperPass());
        // If we didn't decide to inline a function, check to see if we can
        // transform it to pass arguments by value instead of by reference.
        optPM.addPostOrderCGSCCPass(llvm::ArgumentPromotionPass());

        optPM.initFunctionPassManager();
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
        optPM.addFunctionPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
#elif ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        optPM.addFunctionPass(llvm::SROAPass());
#else
        optPM.addFunctionPass(llvm::SROA());
#endif

        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(InstructionSimplifyPass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addFunctionPass(llvm::ReassociatePass());

        // We provide the opt remark emitter pass for LICM to use.
        optPM.addFunctionPass(llvm::RequireAnalysisPass<llvm::OptimizationRemarkEmitterAnalysis, llvm::Function>());

        optPM.setMemorySSA(true);
        optPM.setBlocksFreq(true);
        optPM.initLoopPassManager();
        // Loop passes using MemorySSA
        optPM.addLoopPass(llvm::LoopRotatePass(), 291);
#if ISPC_LLVM_VERSION >= ISPC_LLVM_15_0
        // Use LLVM default options
        llvm::LICMOptions licmOpts;
        optPM.addLoopPass(llvm::LICMPass(licmOpts), 292);
#else
        optPM.addLoopPass(llvm::LICMPass(), 292);
#endif
        if (!g->target->isXeTarget()) {
            // SimpleLoopUnswitch is not a full equivalent of LoopUnswitch pass.
            // It produces much more basic blocks than LoopUnswitch which is
            // not efficient for Xe targets. Moreover when this pass is used
            // some integer division tests are failing on TGLLP Windows.
            // Disable this pass on Xe until the problem is fixed on BE side.
            // Note: enable both trivial and non-trivial loop unswitching.
            optPM.addLoopPass(llvm::SimpleLoopUnswitchPass(true /* NonTrivial */, true /* Trivial */), 293);
        }
        optPM.commitLoopToFunctionPassManager();
        optPM.setMemorySSA(false);
        optPM.setBlocksFreq(false);

        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(InstructionSimplifyPass());

        optPM.initLoopPassManager();
        optPM.addLoopPass(llvm::IndVarSimplifyPass());
        // Currently VC BE does not support memset/memcpy
        // so this pass is temporary disabled for Xe.
        if (!g->target->isXeTarget()) {
            optPM.addLoopPass(llvm::LoopIdiomRecognizePass());
        }

        optPM.addLoopPass(llvm::LoopDeletionPass());
        optPM.commitLoopToFunctionPassManager();

        if (g->opt.unrollLoops) {
            optPM.addFunctionPass(llvm::LoopUnrollPass(), 300);
        }
        // For Xe targets NewGVN pass produces more efficient code due to better resolving of branches.
        // On CPU targets it is effective in optimizing certain types of code,
        // but it is not be beneficial in all cases.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
        if (g->target->isXeTarget()) {
            optPM.addFunctionPass(llvm::NewGVNPass(), 301);
        } else {
            optPM.addFunctionPass(llvm::GVNPass(), 301);
        }
#else
        optPM.addFunctionPass(llvm::GVN(), 301);
#endif
        optPM.addFunctionPass(ReplaceMaskedMemOpsPass());
        optPM.addFunctionPass(IsCompileTimeConstantPass(true));
        optPM.addFunctionPass(IntrinsicsOpt());
        optPM.addFunctionPass(InstructionSimplifyPass());

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget() && g->opt.disableGatherScatterOptimizations == false &&
            g->target->getVectorWidth() > 1) {
            if (!g->opt.disableXeGatherCoalescing) {
                optPM.addFunctionPass(XeGatherCoalescing(), 321);

                // Try the llvm provided load/store vectorizer
                optPM.addFunctionPass(llvm::LoadStoreVectorizerPass(), 325);
            }
        }
#endif
        // Currently VC BE does not support memset/memcpy
        // so this pass is temporary disabled for Xe.
        if (!g->target->isXeTarget()) {
            optPM.addFunctionPass(llvm::MemCpyOptPass());
        }
        optPM.addFunctionPass(llvm::SCCPPass());
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(InstructionSimplifyPass());
        optPM.addFunctionPass(llvm::JumpThreadingPass());
        optPM.addFunctionPass(llvm::CorrelatedValuePropagationPass());
        optPM.addFunctionPass(llvm::DSEPass());
        optPM.addFunctionPass(llvm::ADCEPass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
#if ISPC_LLVM_VERSION >= ISPC_LLVM_18_1
        optPM.addFunctionPass(llvm::InferAlignmentPass());
#endif
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(InstructionSimplifyPass());
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.addFunctionPass(ReplaceLLVMIntrinsics());
        }
#endif

        optPM.addFunctionPass(PeepholePass());
        optPM.addFunctionPass(ScalarizePass());
        optPM.addFunctionPass(llvm::ADCEPass());
        optPM.commitFunctionToModulePassManager();
        optPM.addModulePass(llvm::ModuleInlinerWrapperPass());
        optPM.addModulePass(llvm::StripDeadPrototypesPass());
        optPM.addModulePass(MakeInternalFuncsStaticPass());
        optPM.addModulePass(llvm::GlobalDCEPass());
        optPM.addModulePass(llvm::ConstantMergePass());
#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.initFunctionPassManager();
            optPM.addFunctionPass(CheckIRForXeTarget());
            optPM.addFunctionPass(MangleOpenCLBuiltins());
            optPM.commitFunctionToModulePassManager();
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
