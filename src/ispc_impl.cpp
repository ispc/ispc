/*
  Copyright (c) 2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "args.h"
#include "binary_type.h"
#include "ispc.h"
#include "ispc/ispc.h"
#include "target_registry.h"
#include "type.h"
#include "util.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include <cstring>
#include <llvm/MC/TargetRegistry.h>

namespace ispc {

// Class for managing global state during compilation
class GlobalStateGuard {
  public:
    GlobalStateGuard() : savedModule(m), savedTarget(g->target) {}

    ~GlobalStateGuard() {
        // Restore global state
        m = savedModule;
        g->target = savedTarget;
    }

  private:
    Module *savedModule;
    Target *savedTarget;
};

class ISPCEngine::Impl {
  public:
    Impl() {
        m_output.type = Module::OutputType::Object; // Default output type
    }

    // Fields populated by ParseCommandLineArgs
    std::string m_file;
    Arch m_arch{Arch::none};
    std::string m_cpu;
    std::vector<ISPCTarget> m_targets;
    Module::Output m_output;
    std::vector<std::string> m_linkFileNames;
    bool m_isHelpMode{false};
    bool m_isLinkMode{false};

    bool IsLinkMode() const { return m_isLinkMode; }

    int Compile() {
        // Validate input file
        if (!ValidateInput(m_file)) {
            return 1;
        }
        // Validate output files
        ValidateOutput(m_output);
        if (g->enableTimeTrace) {
            llvm::timeTraceProfilerInitialize(g->timeTraceGranularity, "ispc");
        }

        int ret = 0;
        {
            llvm::TimeTraceScope TimeScope("ExecuteISPCEngine");
            ret = Module::CompileAndOutput(m_file.c_str(), m_arch, m_cpu.empty() ? nullptr : m_cpu.c_str(), m_targets,
                                           m_output);
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

    int Link() {
        std::string filename = !m_output.out.empty() ? m_output.out : "";
        return Module::LinkAndOutput(m_linkFileNames, m_output.type, std::move(filename));
    }

    int Execute() {
        // TODO!: remove the global state.
        GlobalStateGuard guard; // The guard to protect global state

        if (m_isLinkMode) {
            return Link();
        } else if (m_isHelpMode) {
            return 0;
        } else {
            return Compile();
        }
    }

  private:
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

bool Initialize() {
    // Check if already initialized
    if (g != nullptr) {
        return true;
    }

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

    // Initialize globals
    g = new Globals;
    return true;
}

std::unique_ptr<ISPCEngine> ISPCEngine::CreateFromCArgs(int argc, char *argv[]) {
    // Check if library is initialized
    if (g == nullptr) {
        return nullptr;
    }

    auto instance = std::unique_ptr<ISPCEngine>(new ISPCEngine());

    // Parse command line options
    ArgsParseResult parseResult =
        ParseCommandLineArgs(argc, argv, instance->pImpl->m_file, instance->pImpl->m_arch, instance->pImpl->m_cpu,
                             instance->pImpl->m_targets, instance->pImpl->m_output, instance->pImpl->m_linkFileNames,
                             instance->pImpl->m_isLinkMode);

    if (parseResult == ArgsParseResult::failure) {
        return nullptr;
    }

    instance->pImpl->m_isHelpMode = (parseResult == ArgsParseResult::help_requested);

    return instance;
}

std::unique_ptr<ISPCEngine> ISPCEngine::CreateFromArgs(const std::vector<std::string> &args) {
    if (args.empty()) {
        return nullptr;
    }

    // Convert vector to argc/argv format
    int argc = args.size();
    std::vector<std::string> argsCopy(args); // Keep a copy to ensure lifetime
    std::vector<char *> argv(argc);

    for (int i = 0; i < argc; ++i) {
        argv[i] = const_cast<char *>(argsCopy[i].c_str());
    }

    return ISPCEngine::CreateFromCArgs(argc, argv.data());
}

ISPCEngine::ISPCEngine() : pImpl(std::make_unique<Impl>()) {}

ISPCEngine::~ISPCEngine() {}

void Shutdown() {
    // Free all bookkept objects.
    BookKeeper::in().freeAll();

    // Clean up global state
    if (g != nullptr) {
        delete g;
        g = nullptr;
    }
}

int ISPCEngine::Execute() { return pImpl->Execute(); }

// Function that uses C-style argc/argv interface
int CompileFromCArgs(int argc, char *argv[]) {
    // Check if library is initialized
    if (g == nullptr) {
        return 1;
    }

    auto instance = ISPCEngine::CreateFromCArgs(argc, argv);
    if (!instance) {
        return 1;
    }

    return instance->Execute();
}

int CompileFromArgs(const std::vector<std::string> &args) {
    if (args.empty()) {
        return 1;
    }

    // Convert vector to argc/argv format
    int argc = args.size();
    std::vector<std::string> argsCopy(args); // Keep a copy to ensure lifetime
    std::vector<char *> argv(argc);

    for (int i = 0; i < argc; ++i) {
        argv[i] = const_cast<char *>(argsCopy[i].c_str());
    }

    return CompileFromCArgs(argc, argv.data());
}

} // namespace ispc