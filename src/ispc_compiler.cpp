/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ispc_compiler.h"
#include "args.h"
#include "binary_type.h"
#include "ispc.h"
#include "target_registry.h"
#include "type.h"
#include "util.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include <llvm/MC/TargetRegistry.h>

namespace ispc {

class Compiler::Impl {
  public:
    Impl() : m_arch(Arch::none), m_cpu(nullptr), m_isHelpMode(false), m_isLinkMode(false) {
        m_output.type = Module::OutputType::Object; // Default output type
    }

    // Fields populated by ParseCommandLineArgs
    char *m_file = nullptr;
    Arch m_arch;
    const char *m_cpu;
    std::vector<ISPCTarget> m_targets;
    Module::Output m_output;
    std::vector<std::string> m_linkFileNames;
    bool m_isHelpMode;
    bool m_isLinkMode;

    static void writeCompileTimeFile(const char *outFileName) {
        llvm::SmallString<128> jsonFileName(outFileName);
        jsonFileName.append(".json");
        llvm::sys::fs::OpenFlags flags = llvm::sys::fs::OF_Text;
        std::error_code error;
        std::unique_ptr<llvm::ToolOutputFile> of(new llvm::ToolOutputFile(jsonFileName.c_str(), error, flags));

        if (error) {
            Error(SourcePos(), "Cannot open json file \"%s\".\n", jsonFileName.c_str());
            return;
        }

        llvm::raw_fd_ostream &fos(of->os());
        llvm::timeTraceProfilerWrite(fos);
        of->keep();
    }
};

std::unique_ptr<Compiler> Compiler::CreateFromArgs(int argc, char *argv[]) {
    auto driver = std::unique_ptr<Compiler>(new Compiler());

    // Initialize available LLVM targets
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
    // Initialize globals early so we can set option values during parsing.
    g = new Globals;

    // Parse command line options
    ArgsParseResult parseResult = ParseCommandLineArgs(
        argc, argv, driver->pImpl->m_file, driver->pImpl->m_arch, driver->pImpl->m_cpu, driver->pImpl->m_targets,
        driver->pImpl->m_output, driver->pImpl->m_linkFileNames, driver->pImpl->m_isLinkMode);

    if (parseResult == ArgsParseResult::failure) {
        // Clean up global state on failure
        delete g;
        g = nullptr;
        return nullptr;
    }

    driver->pImpl->m_isHelpMode = (parseResult == ArgsParseResult::help_requested);

    return driver;
}

Compiler::Compiler() : pImpl(std::make_unique<Impl>()) {}

Compiler::~Compiler() {
    if (g != nullptr) {
        delete g;
        g = nullptr;
    }
}

void Compiler::Shutdown() {
    // Free all bookkept objects.
    BookKeeper::in().freeAll();
}

bool Compiler::IsLinkMode() const { return pImpl->m_isLinkMode; }

int Compiler::Compile() {
    if (g->enableTimeTrace) {
        llvm::timeTraceProfilerInitialize(g->timeTraceGranularity, "ispc");
    }

    int ret;
    {
        llvm::TimeTraceScope TimeScope("ExecuteCompiler");
        ret = Module::CompileAndOutput(pImpl->m_file, pImpl->m_arch, pImpl->m_cpu, pImpl->m_targets, pImpl->m_output);
    }

    if (g->enableTimeTrace) {
        // Write to file only if compilation is successful.
        if ((ret == 0) && (!pImpl->m_output.out.empty())) {
            Impl::writeCompileTimeFile(pImpl->m_output.out.c_str());
        }
        llvm::timeTraceProfilerCleanup();
    }
    return ret;
}

int Compiler::Link() {
    std::string filename = !pImpl->m_output.out.empty() ? pImpl->m_output.out : "";
    return Module::LinkAndOutput(pImpl->m_linkFileNames, pImpl->m_output.type, filename);
}

int Compiler::Execute() {
    if (pImpl->m_isLinkMode) {
        return Link();
    } else if (pImpl->m_isHelpMode) {
        return 0;
    } else {
        return Compile();
    }
}

} // namespace ispc