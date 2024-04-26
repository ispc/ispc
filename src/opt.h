/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file opt.h
    @brief Declarations related to optimization passes
*/

#pragma once

#include "util.h"

#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/ScopedNoAliasAA.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Target/TargetMachine.h>

namespace ispc {

/** Optimize the functions in the given module, applying the specified
    level of optimization.  optLevel zero corresponds to essentially no
    optimization--just enough to generate correct code, while level one
    corresponds to full optimization.
*/
void Optimize(llvm::Module *module, int optLevel);

///////////////////////////////////////////////////////////////////////////
// This is a wrap over class llvm::ModulePassManager. This duplicates PassManager function run()
//   and adds several add functions with some checks and debug passes.
//   This wrap can control:
//   - If we want to switch off optimization with given number.
//   - If we want to dump LLVM IR after optimization with given number.
//   - If we want to generate LLVM IR debug for gdb after optimization with given number.
class DebugModulePassManager {

  public:
    DebugModulePassManager(llvm::Module &M, int optLevel);
    llvm::PreservedAnalyses run();
    enum Passes { Module, Function, Loop };
    template <typename T> void addModulePass(T &&P, int stage = -1);
    template <typename T> void addPostOrderCGSCCPass(T &&P, int stage = -1);
    template <typename T> void addFunctionPass(T &&P, int stage = -1);
    template <typename T> void addLoopPass(T &&P, int stage = -1);
    // Start a new group of function passes
    void initFunctionPassManager();
    // Add function passes to the ModulePassManager
    void commitFunctionToModulePassManager();
    // Start a new group of loop passes
    void initLoopPassManager();
    // Add loop passes to the FunctionPassManager
    void commitLoopToFunctionPassManager();

    void setMemorySSA(bool v);
    void setBlocksFreq(bool v);

  private:
    llvm::TargetMachine *targetMachine;
    llvm::PassBuilder pb;
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    llvm::ModulePassManager mpm;
    llvm::PassInstrumentationCallbacks PIC;
    llvm::PrintPassOptions PrintPassOpts{/*Verbose*/ true, /*SkipAnalyses*/ true, /*Indent*/ true};
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
    llvm::StandardInstrumentations SI{*g->ctx, /*DebugLogging*/ g->debugPM, /*VerifyEach*/ false, PrintPassOpts};
#else
    llvm::StandardInstrumentations SI{/*DebugLogging*/ g->debugPM, /*VerifyEach*/ false, PrintPassOpts};
#endif
    llvm::OptNoneInstrumentation OptNoneInst{/*DebugLogging*/ false};
    llvm::TimePassesHandler TimePasses{true};

    std::vector<std::unique_ptr<llvm::raw_fd_ostream>> outputDebugDumps;
    std::vector<std::unique_ptr<llvm::FunctionPassManager>> fpmVec;
    std::vector<std::unique_ptr<llvm::LoopPassManager>> lpmVec;

    llvm::Module *m;

    bool m_isFPMOpen{false};
    bool m_isLPMOpen{false};
    bool m_memorySSA{false};
    bool m_blocksFreq{false};
    int m_passNumber;
    int m_optLevel;

    // Add pass to pass manager and print IR if needed
    void addPassAndDebugPrint(std::string name, DebugModulePassManager::Passes kind);
};

template <typename T> void DebugModulePassManager::addModulePass(T &&P, int stage) {
    Assert(!m_isFPMOpen && "FunctionPassManager must be committed before adding module passes.");
    Assert(!m_isLPMOpen && "LoopPassManager must be committed before adding module passes.");
    // taking number of optimization
    m_passNumber = (stage == -1) ? (m_passNumber + 1) : stage;
    if (g->off_stages.find(m_passNumber) == g->off_stages.end()) {
        mpm.addPass(std::move(P));
        addPassAndDebugPrint(T::name().str(), DebugModulePassManager::Passes::Module);
    }
}

template <typename T> void DebugModulePassManager::addPostOrderCGSCCPass(T &&P, int stage) {
    Assert(!m_isFPMOpen && "FunctionPassManager must be committed before adding PostOrderCGSCC passes.");
    Assert(!m_isLPMOpen && "LoopPassManager must be committed before adding PostOrderCGSCC passes.");
    // taking number of optimization
    m_passNumber = (stage == -1) ? (m_passNumber + 1) : stage;
    if (g->off_stages.find(m_passNumber) == g->off_stages.end()) {
        // Add PostOrderCGSCC pass to the ModulePassManager directly through adaptor
        mpm.addPass(llvm::createModuleToPostOrderCGSCCPassAdaptor(std::move(P)));
        addPassAndDebugPrint(T::name().str(), DebugModulePassManager::Passes::Module);
    }
}

template <typename T> void DebugModulePassManager::addFunctionPass(T &&P, int stage) {
    Assert(m_isFPMOpen && "FunctionPassManager must be initialized before adding function passes");
    // taking number of optimization
    m_passNumber = (stage == -1) ? (m_passNumber + 1) : stage;
    if (g->off_stages.find(m_passNumber) == g->off_stages.end()) {
        fpmVec.back()->addPass(std::move(P));
        addPassAndDebugPrint(T::name().str(), DebugModulePassManager::Passes::Function);
    }
}

template <typename T> void DebugModulePassManager::addLoopPass(T &&P, int stage) {
    Assert(m_isLPMOpen && "LoopPassManager must be initialized before adding function passes");
    // taking number of optimization
    m_passNumber = (stage == -1) ? (m_passNumber + 1) : stage;
    if (g->off_stages.find(m_passNumber) == g->off_stages.end()) {
        // if not debug stage, add pass to loop pass manager
        lpmVec.back()->addPass(std::move(P));
        addPassAndDebugPrint(T::name().str(), DebugModulePassManager::Passes::Loop);
    }
}

} // namespace ispc
