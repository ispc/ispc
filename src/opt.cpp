/*
  Copyright (c) 2010-2026, Intel Corporation

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
#include <utility>

#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/Analysis/GlobalsModRef.h>
#include <llvm/Analysis/OptimizationRemarkEmitter.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/Analysis/ProfileSummaryInfo.h>
#include <llvm/Analysis/ScopedNoAliasAA.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRPrinter/IRPrintingPasses.h>
#include <llvm/PassRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/Transforms/IPO/ArgumentPromotion.h>
#include <llvm/Transforms/IPO/ConstantMerge.h>
#include <llvm/Transforms/IPO/DeadArgumentElimination.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/GlobalOpt.h>
#include <llvm/Transforms/IPO/Inliner.h>
#include <llvm/Transforms/IPO/SCCP.h>
#include <llvm/Transforms/IPO/SampleProfile.h>
#include <llvm/Transforms/IPO/StripDeadPrototypes.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#if ISPC_LLVM_VERSION >= ISPC_LLVM_20_0
#include <llvm/Transforms/Utils/Instrumentation.h>
#else
#include <llvm/Transforms/Instrumentation.h>
#endif
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Scalar/CorrelatedValuePropagation.h>
#include <llvm/Transforms/Scalar/DCE.h>
#include <llvm/Transforms/Scalar/DeadStoreElimination.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/IndVarSimplify.h>
#include <llvm/Transforms/Scalar/InferAlignment.h>
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
#include <llvm/Transforms/Vectorize/SLPVectorizer.h>

#ifdef ISPC_XE_ENABLED
#include <llvm/GenXIntrinsics/GenXSPIRVWriterAdaptor.h>
#endif

using namespace ispc;
// Strips all non-alphanumeric characters from given string.
static std::string lSanitize(const std::string &in) {
    std::string res = in;
    llvm::Regex r("[^[:alnum:]]");
    while (r.match(res)) {
        res = r.sub("", res);
    }
    return res;
}

// Get path to dump file
static std::string getDumpFilePath(const std::string &className, int pnum) {
    std::ostringstream filename;
    filename << "ir_" << pnum << "_" << lSanitize(className) << ".ll";

    if (g->dumpFilePath.empty()) {
        return filename.str();
    }

    SourcePos noPos;
    std::error_code EC = llvm::sys::fs::create_directories(g->dumpFilePath);
    if (EC) {
        Error(noPos, "Error creating directory '%s': %s", g->dumpFilePath.c_str(), EC.message().c_str());
    }

    llvm::SmallString<128> pathBuf(g->dumpFilePath);
    llvm::sys::path::append(pathBuf, filename.str());

    return pathBuf.str().str();
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
    pb = llvm::PassBuilder(targetMachine, llvm::PipelineTuningOptions(), std::nullopt, &PIC);

    // Register all the basic analyses with the managers.
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    SI.registerCallbacks(PIC, &mam);

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
        fam.registerPass([aam = std::move(aam)] { return std::move(aam); });
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
                std::string filePath = getDumpFilePath(name, m_passNumber);
                std::unique_ptr<llvm::raw_fd_ostream> outFile =
                    std::make_unique<llvm::raw_fd_ostream>(filePath, EC, llvm::sys::fs::OF_None);
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
#if ISPC_LLVM_VERSION >= ISPC_LLVM_22_0
    // LLVM 22+ removed the BlockFrequencyInfo parameter
    fpmVec.back()->addPass(llvm::createFunctionToLoopPassAdaptor(std::move(*lastLPM), m_memorySSA));
#else
    fpmVec.back()->addPass(llvm::createFunctionToLoopPassAdaptor(std::move(*lastLPM), m_memorySSA, m_blocksFreq));
#endif
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
        optPM.addFunctionPass(llvm::LowerMatrixIntrinsicsPass(), INIT_OPT_NUMBER); // llvm.matrix
        optPM.commitFunctionToModulePassManager();
        optPM.addModulePass(llvm::VerifierPass());
    } else {
        optPM.addModulePass(llvm::VerifierPass(), INIT_OPT_NUMBER);
    }

    optPM.initFunctionPassManager();
    optPM.initLoopPassManager();
    optPM.addLoopPass(llvm::IndVarSimplifyPass());
    optPM.commitLoopToFunctionPassManager();
    optPM.addFunctionPass(LowerISPCIntrinsicsPass(), 11);
    optPM.commitFunctionToModulePassManager();

    llvm::SimplifyCFGOptions simplifyCFGopt;
    simplifyCFGopt.HoistCommonInsts = true;

    // SizeOptLevel of 1 corresponds to the -Os flag and 2 corresponds to the -Oz flag.
    const unsigned SizeOptLevel = (optLevel == 1) ? 1 : 0;
    llvm::InlineParams IP = llvm::getInlineParams(optLevel, SizeOptLevel);
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

        if (g->opt.disableHandlePseudoMemoryOps == false) {
            optPM.addFunctionPass(ReplacePseudoMemoryOpsPass());
        }

        optPM.addFunctionPass(IntrinsicsOpt(), 102);
        optPM.addFunctionPass(IsCompileTimeConstantPass(true));
        optPM.commitFunctionToModulePassManager();

        optPM.addModulePass(llvm::ModuleInlinerWrapperPass(IP));
        optPM.addModulePass(RemovePersistentFuncsPass());

        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addFunctionPass(LowerAMXBuiltinsPass());
        optPM.commitFunctionToModulePassManager();

        optPM.addModulePass(llvm::GlobalDCEPass());

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            optPM.initFunctionPassManager();
            optPM.addFunctionPass(llvm::PromotePass());
            // This pass is needed for correct prints work
            // We don't have any LICM or SimplifyCFG passes scheduled after us, that would cleanup
            // the CFG mess SROAPass may created if allowed to modify CFG, so forbid that.
            optPM.addFunctionPass(llvm::SROAPass(llvm::SROAOptions::PreserveCFG));
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
        optPM.addFunctionPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
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

        // InstCombine contains similar functionality. It can be enabled back
        // (at the moment) with enable-infer-alignment-pass=false option.
        // To preserve previous functionality let's call it before InstCombine every time.
        optPM.addFunctionPass(llvm::InferAlignmentPass());
        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.addFunctionPass(llvm::InstCombinePass(), 210);
            optPM.addFunctionPass(ImproveMemoryOpsPass());
        }
        if (!g->opt.disableMaskAllOnOptimizations) {
            optPM.addFunctionPass(IntrinsicsOpt(), 215);
            optPM.addFunctionPass(InstructionSimplifyPass());
        }
        optPM.addFunctionPass(llvm::DCEPass(), 220);

        // This should be early (before the InstCombinePass and 
        // InstructionSimplifyPass to actually be effective).
        if (g->opt.fastMath != Opt::FastMathMode::None) {
            optPM.addFunctionPass(FastMathPass());
        }

        // On to more serious optimizations
        optPM.addFunctionPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
        optPM.addFunctionPass(llvm::InferAlignmentPass());
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addFunctionPass(llvm::PromotePass());
        optPM.addFunctionPass(llvm::ReassociatePass());
        optPM.setBlocksFreq(true);
        optPM.initLoopPassManager();
        if (g->opt.unrollLoops) {
            // Default optLevel for LoopFullUnrollPass is 2
            optPM.addLoopPass(llvm::LoopFullUnrollPass(optLevel));
        }
        optPM.commitLoopToFunctionPassManager();
        optPM.setBlocksFreq(false);
        optPM.addFunctionPass(ReplaceStdlibShiftPass(), 229);
        optPM.addFunctionPass(llvm::InferAlignmentPass());
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.commitFunctionToModulePassManager();

        optPM.addModulePass(llvm::GlobalOptPass());
        optPM.addModulePass(llvm::IPSCCPPass());
        optPM.addModulePass(llvm::DeadArgumentEliminationPass());

        if (!g->profileSampleUse.empty()) {
            optPM.addModulePass(llvm::SampleProfileLoaderPass(g->profileSampleUse));
            // Cache ProfileSummaryAnalysis as LLVM standard pipeline does
            optPM.addModulePass(llvm::RequireAnalysisPass<llvm::ProfileSummaryAnalysis, llvm::Module>());
        }

        //  No such pass with new PM
        //  https://reviews.llvm.org/D44415
        //  optPM.add(llvm::createPruneEHPass());
        optPM.addPostOrderCGSCCPass(llvm::PostOrderFunctionAttrsPass());
        optPM.addModulePass(llvm::ReversePostOrderFunctionAttrsPass());

        // Next inline pass will remove functions, saved by __keep_funcs_live
        optPM.addModulePass(llvm::ModuleInlinerWrapperPass(IP));

        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::InstSimplifyPass());
        optPM.addFunctionPass(llvm::DCEPass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addFunctionPass(llvm::ADCEPass());
        optPM.addFunctionPass(llvm::InferAlignmentPass());
        optPM.addFunctionPass(ReplaceMaskedMemOpsPass());
        optPM.addFunctionPass(llvm::InstCombinePass(), 241);
        optPM.addFunctionPass(llvm::JumpThreadingPass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addFunctionPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
        optPM.addFunctionPass(llvm::InferAlignmentPass());
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.commitFunctionToModulePassManager();

#ifdef ISPC_XE_ENABLED
        if (g->target->isXeTarget()) {
            // Inline
            optPM.initFunctionPassManager();
            optPM.addFunctionPass(llvm::CorrelatedValuePropagationPass());
            optPM.addFunctionPass(llvm::InferAlignmentPass());
            optPM.addFunctionPass(llvm::InstCombinePass());
            optPM.commitFunctionToModulePassManager();
            optPM.addModulePass(llvm::GlobalDCEPass());
            optPM.initFunctionPassManager();
            optPM.addFunctionPass(llvm::InferAlignmentPass());
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
            optPM.addFunctionPass(llvm::InferAlignmentPass());
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

        optPM.addModulePass(llvm::ModuleInlinerWrapperPass(IP), 265);
        // If we didn't decide to inline a function, check to see if we can
        // transform it to pass arguments by value instead of by reference.
        optPM.addPostOrderCGSCCPass(llvm::ArgumentPromotionPass());

        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::InstSimplifyPass());
        optPM.addFunctionPass(IntrinsicsOpt());
        optPM.addFunctionPass(InstructionSimplifyPass());

        if (g->opt.disableGatherScatterOptimizations == false && g->target->getVectorWidth() > 1) {
            optPM.addFunctionPass(llvm::InferAlignmentPass());
            optPM.addFunctionPass(llvm::InstCombinePass(), 270);
            optPM.addFunctionPass(ImproveMemoryOpsPass());
        }
        optPM.commitFunctionToModulePassManager();
        optPM.addModulePass(llvm::IPSCCPPass(), 275);
        optPM.addModulePass(llvm::DeadArgumentEliminationPass());

        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::ADCEPass());
        optPM.addFunctionPass(llvm::InferAlignmentPass());
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));

        if (g->opt.disableHandlePseudoMemoryOps == false) {
            optPM.addFunctionPass(ReplacePseudoMemoryOpsPass(), 280);
        }
        optPM.addFunctionPass(IntrinsicsOpt(), 281);
        optPM.addFunctionPass(InstructionSimplifyPass());
        optPM.commitFunctionToModulePassManager();

        optPM.addModulePass(llvm::ModuleInlinerWrapperPass(IP));
        // If we didn't decide to inline a function, check to see if we can
        // transform it to pass arguments by value instead of by reference.
        optPM.addPostOrderCGSCCPass(llvm::ArgumentPromotionPass());

        optPM.initFunctionPassManager();
        optPM.addFunctionPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
        optPM.addFunctionPass(llvm::InferAlignmentPass());
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
        // Use LLVM default options
        llvm::LICMOptions licmOpts;
        optPM.addLoopPass(llvm::LICMPass(licmOpts), 292);
        if (!g->target->isXeTarget()) {
            // SimpleLoopUnswitch is not a full equivalent of LoopUnswitch pass.
            // It produces much more basic blocks than LoopUnswitch which is
            // not efficient for Xe targets. Moreover when this pass is used
            // some integer division tests are failing on TGLLP Windows.
            // Disable this pass on Xe until the problem is fixed on BE side.
            // Note: enable both trivial and non-trivial loop unswitching.
            optPM.addLoopPass(llvm::SimpleLoopUnswitchPass(optLevel > 1 /* NonTrivial */, true /* Trivial */), 293);
        }
        optPM.commitLoopToFunctionPassManager();
        optPM.setMemorySSA(false);
        optPM.setBlocksFreq(false);
        optPM.addFunctionPass(llvm::InferAlignmentPass());
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
            // Default optLevel for LoopUnrollPass is 2
            optPM.addFunctionPass(llvm::LoopUnrollPass(optLevel), 300);
        }
        // For Xe targets NewGVN pass produces more efficient code due to better resolving of branches.
        // On CPU targets it is effective in optimizing certain types of code,
        // but it is not be beneficial in all cases.
        if (g->target->isXeTarget()) {
            optPM.addFunctionPass(llvm::NewGVNPass(), 301);
        } else {
            optPM.addFunctionPass(llvm::GVNPass(), 301);
        }
        optPM.addFunctionPass(ReplaceMaskedMemOpsPass());
        optPM.addFunctionPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
        optPM.addFunctionPass(llvm::InferAlignmentPass());
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(IsCompileTimeConstantPass(true));
        optPM.addFunctionPass(LowerAMXBuiltinsPass());
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
        // On ARM NEON targets LoadStoreVectorizer works great to combine
        // multiple loads/stores into a single vector load/store as in #2052.
        // However because of LoadStoreVectorizer after vector loads,
        // elements are often extracted and processed individually using scalar operations.
        // To vectorize these scalar operations, SLP vectorizer is used.
        // Enabling it on other targets may be beneficial but require extensive testing.
        if (ISPCTargetIsNeon(g->target->getISPCTarget()) || g->opt.enableLoadStoreVectorizer) {
            optPM.addFunctionPass(llvm::LoadStoreVectorizerPass());
        }
        if (ISPCTargetIsNeon(g->target->getISPCTarget()) || g->opt.enableSLPVectorizer) {
            optPM.addFunctionPass(llvm::SLPVectorizerPass());
        }
        // Currently VC BE does not support memset/memcpy
        // so this pass is temporary disabled for Xe.
        if (!g->target->isXeTarget()) {
            optPM.addFunctionPass(llvm::MemCpyOptPass());
        }
        optPM.addFunctionPass(llvm::SCCPPass());
        optPM.addFunctionPass(llvm::InferAlignmentPass());
        optPM.addFunctionPass(llvm::InstCombinePass());
        optPM.addFunctionPass(InstructionSimplifyPass());
        optPM.addFunctionPass(llvm::JumpThreadingPass());
        optPM.addFunctionPass(llvm::CorrelatedValuePropagationPass());
        optPM.addFunctionPass(llvm::DSEPass());
        optPM.addFunctionPass(llvm::ADCEPass());
        optPM.addFunctionPass(llvm::SimplifyCFGPass(simplifyCFGopt));
        optPM.addFunctionPass(llvm::InferAlignmentPass());
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
        optPM.addModulePass(llvm::ModuleInlinerWrapperPass(IP));
        optPM.addModulePass(llvm::StripDeadPrototypesPass());
        optPM.addModulePass(RemovePersistentFuncsPass());
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
