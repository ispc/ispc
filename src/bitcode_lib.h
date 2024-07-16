/*
  Copyright (c) 2019-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file bitcode_lib.h
    @brief a header to host BitcodeLib - a wrapper for single bitcode library.
*/

#pragma once

#include "target_enums.h"

#include <llvm/IR/Module.h>

namespace ispc {

class BitcodeLib {
  public:
    enum class BitcodeLibType { Dispatch, Builtins_c, ISPC_target, Stdlib };
    enum class BitcodeLibStorage { FileSystem, Embedded };

  private:
    // Type of library
    BitcodeLibType m_type;
    BitcodeLibStorage m_storage;

    // The code and its size
    const unsigned char *m_lib;
    const size_t m_size;

    // Identification of the library: OS, Arch, ISPCTarget
    const TargetOS m_os;
    const Arch m_arch;
    const ISPCTarget m_target;

    const std::string m_filename;

  public:
    // Every constructor is presented in two types: one for embedded bitcode
    // library and one for file system bitcode.
    // Dispatch constructor
    BitcodeLib(const unsigned char lib[], int size, TargetOS os);
    BitcodeLib(const char *filename, TargetOS os);
    // Builtins-c constructor
    BitcodeLib(const unsigned char lib[], int size, TargetOS os, Arch arch);
    BitcodeLib(const char *filename, TargetOS os, Arch arch);
    // ISPC-target constructor
    BitcodeLib(const unsigned char lib[], int size, ISPCTarget target, TargetOS os, Arch arch);
    BitcodeLib(const char *filename, ISPCTarget target, TargetOS os, Arch arch);
    // General constructor
    BitcodeLib(BitcodeLibType type, const unsigned char lib[], int size, ISPCTarget target, TargetOS os, Arch arch);
    BitcodeLib(BitcodeLibType type, const char *filename, ISPCTarget target, TargetOS os, Arch arch);
    void print() const;

    BitcodeLibType getType() const;
    const unsigned char *getLib() const;
    size_t getSize() const;
    TargetOS getOS() const;
    Arch getArch() const;
    ISPCTarget getISPCTarget() const;
    const std::string &getFilename() const;
    bool fileExists() const;
    llvm::Module *getLLVMModule() const;
};

} // namespace ispc
