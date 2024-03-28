/*
  Copyright (c) 2019-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file bitcode_lib.cpp
    @brief BitcodeLib represents single bitcode library file.
*/

#include "bitcode_lib.h"
#include "ispc.h"
#include "target_registry.h"

#include "llvm/Support/MemoryBuffer.h"
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

using namespace ispc;

// Dispatch constructor
BitcodeLib::BitcodeLib(const char *filename, TargetOS os)
    : m_os(os), m_arch(Arch::none), m_target(ISPCTarget::none), m_filename(filename) {}

// Builtins-c constructor
BitcodeLib::BitcodeLib(const char *filename, TargetOS os, Arch arch)
    : m_os(os), m_arch(arch), m_target(ISPCTarget::none), m_filename(filename) {}

BitcodeLib::BitcodeLib(const char *filename, ISPCTarget target, TargetOS os, Arch arch)
    : m_os(os), m_arch(arch), m_target(target), m_filename(filename) {}

// TODO: this is debug version: either remove or make it use friendly.
void BitcodeLib::print() const {
    std::string os = OSToString(m_os);
    std::string target = ISPCTargetToString(m_target);
    std::string arch = ArchToString(m_arch);
    printf("OS: %s, target: %s, arch(runtime) %s, filename: %s\n", os.c_str(), target.c_str(), arch.c_str(),
           m_filename.c_str());
}

TargetOS BitcodeLib::getOS() const { return m_os; }
Arch BitcodeLib::getArch() const { return m_arch; }
ISPCTarget BitcodeLib::getISPCTarget() const { return m_target; }
const std::string &BitcodeLib::getFilename() const { return m_filename; }

bool BitcodeLib::fileExists() const {
    llvm::SmallString<128> filePath(g->shareDirPath);
    llvm::sys::path::append(filePath, m_filename);
    return llvm::sys::fs::exists(filePath);
}

llvm::Module *BitcodeLib::getLLVMModule() const {
    llvm::SmallString<128> filePath(g->shareDirPath);
    llvm::sys::path::append(filePath, m_filename);
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrErr = llvm::MemoryBuffer::getFile(filePath.str());
    // TODO! proper exit when load module fails
    if (std::error_code EC = bufferOrErr.getError()) {
        fprintf(stderr, "ERROR reading bc_filename %s\n", m_filename.c_str());
        fprintf(stderr, "%s\n", EC.message().c_str());
    }
    llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
        llvm::parseBitcodeFile(bufferOrErr->get()->getMemBufferRef(), *g->ctx);
    if (!ModuleOrErr) {
        fprintf(stderr, "ERROR parsing bc_filename %s\n", m_filename.c_str());
    } else {
        llvm::Module *M = ModuleOrErr.get().release();
        return M;
    }
    return nullptr;
}
