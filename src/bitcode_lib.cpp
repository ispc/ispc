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

/** @file bitcode_lib.cpp
    @brief BitcodeLib represents single bitcode library file (either dispatch,
           Builtiins-c, or ISPCTarget).
*/

#include "bitcode_lib.h"
#include "target_registry.h"

using namespace ispc;

// Dispatch constructor
BitcodeLib::BitcodeLib(const unsigned char lib[], int size, TargetOS os)
    : m_type(BitcodeLibType::Dispatch), m_lib(lib), m_size(size), m_os(os), m_arch(Arch::none),
      m_target(ISPCTarget::none) {
    TargetLibRegistry::RegisterTarget(this);
}
// Builtins-c constructor
BitcodeLib::BitcodeLib(const unsigned char lib[], int size, TargetOS os, Arch arch)
    : m_type(BitcodeLibType::Builtins_c), m_lib(lib), m_size(size), m_os(os), m_arch(arch), m_target(ISPCTarget::none) {
    TargetLibRegistry::RegisterTarget(this);
}
// ISPC-target constructor
BitcodeLib::BitcodeLib(const unsigned char lib[], int size, ISPCTarget target, TargetOS os, Arch arch)
    : m_type(BitcodeLibType::ISPC_target), m_lib(lib), m_size(size), m_os(os), m_arch(arch), m_target(target) {
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
    }
}

BitcodeLib::BitcodeLibType BitcodeLib::getType() const { return m_type; }
const unsigned char *BitcodeLib::getLib() const { return m_lib; }
const size_t BitcodeLib::getSize() const { return m_size; }
const TargetOS BitcodeLib::getOS() const { return m_os; }
const Arch BitcodeLib::getArch() const { return m_arch; }
const ISPCTarget BitcodeLib::getISPCTarget() const { return m_target; }
