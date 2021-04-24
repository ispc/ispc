/*
  Copyright (c) 2019-2021, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
    const size_t getSize() const;
    const TargetOS getOS() const;
    const Arch getArch() const;
    const ISPCTarget getISPCTarget() const;
};

} // namespace ispc
