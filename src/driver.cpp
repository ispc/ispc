/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "driver.h"
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

std::unique_ptr<Driver> Driver::CreateFromArgs(int argc, char *argv[]) {
    auto driver = std::unique_ptr<Driver>(new Driver());

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
    ArgsParseResult parseResult =
        ParseCommandLineArgs(argc, argv, driver->m_file, driver->m_arch, driver->m_cpu, driver->m_targets,
                             driver->m_output, driver->m_linkFileNames, driver->m_isLinkMode);

    if (parseResult == ArgsParseResult::failure) {
        // Clean up global state on failure
        delete g;
        g = nullptr;
        return nullptr;
    }

    driver->m_isHelpMode = (parseResult == ArgsParseResult::help_requested);

    return driver;
}

Driver::Driver() : m_arch(Arch::none), m_cpu(nullptr), m_isHelpMode(false), m_isLinkMode(false) {
    m_output.type = Module::OutputType::Object; // Default output type
}

Driver::~Driver() {
    if (g != nullptr) {
        delete g;
        g = nullptr;
    }
}

void Driver::Shutdown() {
    // Free all bookkept objects.
    BookKeeper::in().freeAll();
}

bool Driver::IsLinkMode() const { return m_isLinkMode; }

int Driver::Compile() {
    if (g->enableTimeTrace) {
        llvm::timeTraceProfilerInitialize(g->timeTraceGranularity, "ispc");
    }

    int ret;
    {
        llvm::TimeTraceScope TimeScope("ExecuteCompiler");
        ret = Module::CompileAndOutput(m_file, m_arch, m_cpu, m_targets, m_output);
    }

    if (g->enableTimeTrace) {
        // Write to file only if compilation is successful.
        if ((ret == 0) && (!m_output.out.empty())) {
            writeCompileTimeFile(m_output.out.c_str());
        }
        llvm::timeTraceProfilerCleanup();
    }
    return ret;
}

int Driver::Link() {
    std::string filename = !m_output.out.empty() ? m_output.out : "";
    return Module::LinkAndOutput(m_linkFileNames, m_output.type, filename);
}

int Driver::Execute() {
    if (m_isLinkMode) {
        return Link();
    } else if (m_isHelpMode) {
        return 0;
    } else {
        return Compile();
    }
}

void Driver::writeCompileTimeFile(const char *outFileName) {
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

} // namespace ispc