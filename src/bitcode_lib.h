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
  private:
    // Identification of the library: OS, Arch, ISPCTarget
    const TargetOS m_os;
    const Arch m_arch;
    const ISPCTarget m_target;

    const std::string m_filename;

  public:
    // Dispatch constructor
    BitcodeLib(const char *filename, TargetOS os);
    // Builtins-c constructor
    BitcodeLib(const char *filename, TargetOS os, Arch arch);
    // ISPC-target constructor
    BitcodeLib(const char *filename, ISPCTarget target, TargetOS os, Arch arch);
    void print() const;

    TargetOS getOS() const;
    Arch getArch() const;
    ISPCTarget getISPCTarget() const;
    const std::string &getFilename() const;
    bool fileExists() const;
    llvm::Module *getLLVMModule() const;
};

} // namespace ispc
