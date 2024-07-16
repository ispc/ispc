/*
  Copyright (c) 2019-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file bitcode_lib.cpp
    @brief BitcodeLib represents single bitcode library file (either dispatch,
           Builtiins-c, or ISPCTarget).
*/

#include <stdlib.h>

#include "bitcode_lib.h"
#include "ispc.h"
#include "target_registry.h"
#include "util.h"

#include "llvm/Support/MemoryBuffer.h"
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

using namespace ispc;

// Dispatch constructors
BitcodeLib::BitcodeLib(const unsigned char lib[], int size, TargetOS os)
    : m_type(BitcodeLibType::Dispatch), m_storage(BitcodeLibStorage::Embedded), m_lib(lib), m_size(size), m_os(os),
      m_arch(Arch::none), m_target(ISPCTarget::none) {
    TargetLibRegistry::RegisterTarget(this);
}

BitcodeLib::BitcodeLib(const char *filename, TargetOS os)
    : m_type(BitcodeLibType::Dispatch), m_storage(BitcodeLibStorage::FileSystem), m_lib(nullptr), m_size(0), m_os(os),
      m_arch(Arch::none), m_target(ISPCTarget::none), m_filename(filename) {
    TargetLibRegistry::RegisterTarget(this);
}

// Builtins-c constructors
BitcodeLib::BitcodeLib(const unsigned char lib[], int size, TargetOS os, Arch arch)
    : m_type(BitcodeLibType::Builtins_c), m_storage(BitcodeLibStorage::Embedded), m_lib(lib), m_size(size), m_os(os),
      m_arch(arch), m_target(ISPCTarget::none) {
    TargetLibRegistry::RegisterTarget(this);
}

BitcodeLib::BitcodeLib(const char *filename, TargetOS os, Arch arch)
    : m_type(BitcodeLibType::Builtins_c), m_storage(BitcodeLibStorage::FileSystem), m_lib(nullptr), m_size(0), m_os(os),
      m_arch(arch), m_target(ISPCTarget::none), m_filename(filename) {
    TargetLibRegistry::RegisterTarget(this);
}

// ISPC-target constructors
BitcodeLib::BitcodeLib(const unsigned char lib[], int size, ISPCTarget target, TargetOS os, Arch arch)
    : m_type(BitcodeLibType::ISPC_target), m_storage(BitcodeLibStorage::Embedded), m_lib(lib), m_size(size), m_os(os),
      m_arch(arch), m_target(target) {
    TargetLibRegistry::RegisterTarget(this);
}

BitcodeLib::BitcodeLib(const char *filename, ISPCTarget target, TargetOS os, Arch arch)
    : m_type(BitcodeLibType::ISPC_target), m_storage(BitcodeLibStorage::FileSystem), m_lib(nullptr), m_size(0),
      m_os(os), m_arch(arch), m_target(target), m_filename(filename) {
    TargetLibRegistry::RegisterTarget(this);
}

// General constructors
BitcodeLib::BitcodeLib(BitcodeLibType type, const unsigned char lib[], int size, ISPCTarget target, TargetOS os,
                       Arch arch)
    : m_type(type), m_storage(BitcodeLibStorage::Embedded), m_lib(lib), m_size(size), m_os(os), m_arch(arch),
      m_target(target) {
    TargetLibRegistry::RegisterTarget(this);
}

BitcodeLib::BitcodeLib(BitcodeLibType type, const char *filename, ISPCTarget target, TargetOS os, Arch arch)
    : m_type(type), m_storage(BitcodeLibStorage::FileSystem), m_lib(nullptr), m_size(0), m_os(os), m_arch(arch),
      m_target(target), m_filename(filename) {
    TargetLibRegistry::RegisterTarget(this);
}

// TODO: this is debug version: either remove or make it use friendly.
void BitcodeLib::print() const {
    std::string os = OSToString(m_os);
    switch (m_type) {
    case BitcodeLibType::Dispatch: {
        printf("Type: dispatch.    size: %zu, OS: %s\n", m_size, os.c_str());
        break;
    }
    case BitcodeLibType::Builtins_c: {
        std::string arch = ArchToString(m_arch);
        printf("Type: builtins-c.  size: %zu, OS: %s, arch: %s\n", m_size, os.c_str(), arch.c_str());
        break;
    }
    case BitcodeLibType::ISPC_target: {
        std::string target = ISPCTargetToString(m_target);
        std::string arch = ArchToString(m_arch);
        printf("Type: ispc-target. size: %zu, OS: %s, target: %s, arch(runtime) %s\n", m_size, os.c_str(),
               target.c_str(), arch.c_str());
        break;
    }
    case BitcodeLibType::Stdlib: {
        std::string target = ISPCTargetToString(m_target);
        std::string arch = ArchToString(m_arch);
        printf("Type: stdlib.      size: %zu, OS: %s, target: %s, arch(runtime) %s\n", m_size, os.c_str(),
               target.c_str(), arch.c_str());
        break;
    }
    }
}

BitcodeLib::BitcodeLibType BitcodeLib::getType() const { return m_type; }
const unsigned char *BitcodeLib::getLib() const { return m_lib; }
size_t BitcodeLib::getSize() const { return m_size; }
TargetOS BitcodeLib::getOS() const { return m_os; }
Arch BitcodeLib::getArch() const { return m_arch; }
ISPCTarget BitcodeLib::getISPCTarget() const { return m_target; }
const std::string &BitcodeLib::getFilename() const { return m_filename; }

bool BitcodeLib::fileExists() const {
    llvm::SmallString<128> filePath(g->shareDirPath);
    llvm::sys::path::append(filePath, m_filename);
    if (m_filename.empty()) {
        // If filename is empty, it means that the library is embedded.
        // So both true and false are meaningless, so return true to not fill
        // the missedFiles in printSupportMatrix.
        return true;
    }
    return llvm::sys::fs::exists(filePath);
}

llvm::Module *BitcodeLib::getLLVMModule() const {
    switch (m_storage) {
    case BitcodeLibStorage::FileSystem: {
        llvm::SmallString<128> filePath(g->shareDirPath);
        llvm::sys::path::append(filePath, m_filename);
        llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrErr = llvm::MemoryBuffer::getFile(filePath.str());
        if (std::error_code EC = bufferOrErr.getError()) {
            Error(SourcePos(), "Error reading bc_filename %s\n%s\n", m_filename.c_str(), EC.message().c_str());
            exit(1);
        }
        llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
            llvm::parseBitcodeFile(bufferOrErr->get()->getMemBufferRef(), *g->ctx);
        if (!ModuleOrErr) {
            Error(SourcePos(), "Error parsing bitcode from filename %s\n", m_filename.c_str());
            exit(1);
        } else {
            llvm::Module *M = ModuleOrErr.get().release();
            return M;
        }
        return nullptr;
    }
    case BitcodeLibStorage::Embedded: {
        llvm::StringRef sb = llvm::StringRef((const char *)m_lib, m_size);
        llvm::MemoryBufferRef bcBuf = llvm::MemoryBuffer::getMemBuffer(sb)->getMemBufferRef();
        llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr = llvm::parseBitcodeFile(bcBuf, *g->ctx);
        if (!ModuleOrErr) {
            Error(SourcePos(), "Error parsing stdlib bitcode: %s", toString(ModuleOrErr.takeError()).c_str());
            exit(1);
        } else {
            llvm::Module *M = ModuleOrErr.get().release();
            return M;
        }
        return nullptr;
    }
    default:
        Error(SourcePos(), "Error loading bitcode library\n");
        exit(1);
    }
}
