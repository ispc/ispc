/*
  Copyright (c) 2019-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file bitcode_lib.h
    @brief a header to host BitcodeLib - a wrapper for single bitcode library.
*/

#pragma once

#include "target_enums.h"

namespace ispc {

class BitcodeLib {
  public:
    enum class BitcodeLibType { Dispatch, Builtins_c, ISPC_target };

  private:
    // Type of library
    BitcodeLibType m_type;

    // The code and its size
    const unsigned char *m_lib;
    const size_t m_size;

    // Identification of the library: OS, Arch, ISPCTarget
    const TargetOS m_os;
    const Arch m_arch;
    const ISPCTarget m_target;

  public:
    // Dispatch constructor
    BitcodeLib(const unsigned char lib[], int size, TargetOS os);
    // Builtins-c constructor
    BitcodeLib(const unsigned char lib[], int size, TargetOS os, Arch arch);
    // ISPC-target constructor
    BitcodeLib(const unsigned char lib[], int size, ISPCTarget target, TargetOS os, Arch arch);
    void print() const;

    BitcodeLibType getType() const;
    const unsigned char *getLib() const;
    size_t getSize() const;
    TargetOS getOS() const;
    Arch getArch() const;
    ISPCTarget getISPCTarget() const;
};

} // namespace ispc
