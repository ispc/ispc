/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ispc.h"
#include "llvmutil.h"
#include "opt.h"
#include "src/opt/ISPCPasses.h"
#include "target_enums.h"
#include "util.h"

#include "llvm/Support/CommandLine.h"
#include <llvm/Support/Signals.h>
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
#include <llvm/IRPrinter/IRPrintingPasses.h>
#else
#include <llvm/IR/IRPrintingPasses.h>
#endif
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>

using namespace llvm;

static cl::opt<std::string> Passes("passes", cl::desc("Passes to run"));
static cl::opt<bool> PrintPasses("print-passes", cl::desc("Print passes"));
static cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input bitcode file>"), cl::init("-"),
                                          cl::value_desc("filename"));
static cl::opt<std::string> OutputFilename("o", cl::desc("Override output filename"), cl::value_desc("filename"));
static cl::opt<std::string> TargetTarget("target", cl::desc("ISPC target"), cl::init("host"), cl::value_desc("target"));
// TODO: unused for now.
static cl::opt<std::string> TargetArch("arch", cl::desc("ISPC target architecture"), cl::value_desc("arch"));
static cl::opt<std::string> TargetCPU("cpu", cl::desc("ISPC target CPU"), cl::value_desc("cpu"));
// TODO: target-os

static void lSignal(void *) {
    using namespace ispc;
    FATAL("Unhandled signal sent to process; terminating.");
}

static void lPrintPassName(StringRef PassName, raw_ostream &OS) { OS << "  " << PassName << "\n"; }

static void lPrintPasses(raw_ostream &OS) {
    OS << "Module passes:\n";
#define MODULE_PASS(NAME, CREATE_PASS) lPrintPassName(NAME, OS);
#include "opt/ISPCPassRegistry.def"

    OS << "Function passes:\n";
#define FUNCTION_PASS(NAME, CREATE_PASS) lPrintPassName(NAME, OS);
#include "opt/ISPCPassRegistry.def"
}

static void lAddPass(ispc::DebugModulePassManager &PM, const std::string &PassName) {
    using namespace ispc;
#define MODULE_PASS(NAME, CREATE_PASS)                                                                                 \
    if (PassName == NAME) {                                                                                            \
        PM.addModulePass(CREATE_PASS);                                                                                 \
    }
#define FUNCTION_PASS(NAME, CREATE_PASS)                                                                               \
    if (PassName == NAME) {                                                                                            \
        PM.initFunctionPassManager();                                                                                  \
        PM.addFunctionPass(CREATE_PASS);                                                                               \
        PM.commitFunctionToModulePassManager();                                                                        \
    }
#include "opt/ISPCPassRegistry.def"
}

int main(int argc, char **argv) {
    llvm::sys::AddSignalHandler(lSignal, nullptr);
    // initialize available LLVM targets
#ifdef ISPC_X86_ENABLED
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();
    LLVMInitializeX86TargetMC();
#endif

#ifdef ISPC_ARM_ENABLED
    LLVMInitializeARMTargetInfo();
    LLVMInitializeARMTarget();
    LLVMInitializeARMAsmPrinter();
    LLVMInitializeARMAsmParser();
    LLVMInitializeARMDisassembler();
    LLVMInitializeARMTargetMC();

    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64AsmPrinter();
    LLVMInitializeAArch64AsmParser();
    LLVMInitializeAArch64Disassembler();
    LLVMInitializeAArch64TargetMC();
#endif

#ifdef ISPC_WASM_ENABLED
    LLVMInitializeWebAssemblyAsmParser();
    LLVMInitializeWebAssemblyAsmPrinter();
    LLVMInitializeWebAssemblyDisassembler();
    LLVMInitializeWebAssemblyTarget();
    LLVMInitializeWebAssemblyTargetInfo();
    LLVMInitializeWebAssemblyTargetMC();
#endif

    cl::ParseCommandLineOptions(argc, argv);

    if (PrintPasses) {
        lPrintPasses(outs());
        return 0;
    }

    ispc::g = new ispc::Globals;
    LLVMContext *ctx = ispc::g->ctx;

    ispc::ISPCTarget target = ispc::ParseISPCTarget(TargetTarget);

    // TODO: here, we rely on arch and cpu autodetection in ispc::Target constructor.
    ispc::g->target =
        new ispc::Target(ispc::Arch::none, nullptr, target, ispc::PICLevel::NotPIC, ispc::MCModel::Default, false);
    if (!ispc::g->target->isValid()) {
        ispc::Error(ispc::SourcePos(), "Unsupported target\n");
        return 1;
    }

    // If it is true then LLVM Assembly won't be read.
    ctx->setDiscardValueNames(false);

    ispc::InitLLVMUtil(ctx, *ispc::g->target);

    if (Passes.empty()) {
        ispc::Error(ispc::SourcePos(), "No pass specified");
        return 1;
    }

    SMDiagnostic err;
    std::error_code EC;

    auto M = getLazyIRFileModule(InputFilename, err, *ctx);
    if (!M.get()) {
        err.print(argv[0], errs());
        return 1;
    }

    ToolOutputFile Out(OutputFilename, EC, sys::fs::OF_None);
    if (EC) {
        ispc::Error(ispc::SourcePos(), "Error opening output file: %s: %s", OutputFilename.c_str(),
                    EC.message().c_str());
        return 1;
    }

    ispc::DebugModulePassManager PM(*M, 0);

    // TODO: support multiple passes separated by comma.
    lAddPass(PM, Passes);
    PM.addModulePass(PrintModulePass(Out.os(), ""));

    PM.run();
    Out.keep();

    return 0;
}
