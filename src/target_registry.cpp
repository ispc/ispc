/*
  Copyright (c) 2019, Intel Corporation
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

/** @file target_registry.h
    @brief Registry to handle bitcode libraries.
*/

#include "target_registry.h"

// Returns number of bits required to store this value
static constexpr uint32_t bits_required(uint32_t x) {
    int msb_position = 0;
    while (x > 0) {
        x = x >> 1;
        msb_position++;
    }
    return msb_position;
}

class Triple {
    // Encode values |0000|arch|os|target|
    static constexpr uint32_t target_width = bits_required((uint32_t)ISPCTarget::error);
    static constexpr uint32_t os_width = bits_required((uint32_t)TargetOS::error);
    static constexpr uint32_t arch_width = bits_required((uint32_t)Arch::error);
    static_assert(target_width + os_width + arch_width <= 32, "Too large value to encode");
    static constexpr uint32_t target_mask = (1 << target_width) - 1;
    static constexpr uint32_t os_mask = ((1 << os_width) - 1) << target_width;
    static constexpr uint32_t arch_mask = ((1 << arch_width) - 1) << (target_width + os_width);

  public:
    ISPCTarget m_target;
    TargetOS m_os;
    Arch m_arch;

    Triple(uint32_t encoding) {
        m_target = (ISPCTarget)(encoding & target_mask);
        m_os = (TargetOS)((encoding & os_mask) >> target_width);
        m_arch = (Arch)((encoding & arch_mask) >> (target_width + os_width));
    };

    Triple(ISPCTarget target, TargetOS os, Arch arch) : m_target(target), m_os(os), m_arch(arch){};

    uint32_t encode() const {
        uint32_t result = (uint32_t)m_arch;
        result = (result << os_width) + (uint32_t)m_os;
        result = (result << target_width) + (uint32_t)m_target;
        return result;
    };
};

std::vector<const BitcodeLib *> *TargetLibRegistry::libs = nullptr;

TargetLibRegistry::TargetLibRegistry() {
    // TODO: sort before adding - to canonicalize.
    // TODO: check for conflicts / duplicates.
    for (auto lib : *libs) {
        switch (lib->getType()) {
        case BitcodeLib::BitcodeLibType::Dispatch:
            m_dispatch = lib;
            break;
        case BitcodeLib::BitcodeLibType::Builtins_c:
            m_builtins[Triple(lib->getISPCTarget(), lib->getOS(), lib->getArch()).encode()] = lib;
            break;
        case BitcodeLib::BitcodeLibType::ISPC_target:
            m_targets[Triple(lib->getISPCTarget(), lib->getOS(), lib->getArch()).encode()] = lib;
            break;
        }
    }
}

void TargetLibRegistry::RegisterTarget(const BitcodeLib *lib) {
    if (!libs) {
        libs = new std::vector<const BitcodeLib *>();
    }
    libs->push_back(lib);
}

TargetLibRegistry *TargetLibRegistry::getTargetLibRegistry() {
    static TargetLibRegistry *reg = new TargetLibRegistry();
    return reg;
}

const BitcodeLib *TargetLibRegistry::getDispatchLib() const { return m_dispatch; }

const BitcodeLib *TargetLibRegistry::getBuiltinsCLib(TargetOS os, Arch arch) const {
    auto result = m_builtins.find(Triple(ISPCTarget::none, os, arch).encode());
    if (result != m_builtins.end()) {
        return result->second;
    }
    return nullptr;
}
const BitcodeLib *TargetLibRegistry::getISPCTargetLib(ISPCTarget target, TargetOS os, Arch arch) const {
    // TODO: validate parameters not to be errors or forbidden values.

    // This is an alias. It might be a good idea generalize this.
    if (target == ISPCTarget::avx1_i32x4) {
        target = ISPCTarget::sse4_i32x4;
    }

    // Canonicalize OS, as for the target we only differentiate between Windows and Unix.
    os = (os == TargetOS::windows) ? TargetOS::windows : TargetOS::linux;
    auto result = m_targets.find(Triple(target, os, arch).encode());
    if (result != m_targets.end()) {
        return result->second;
    }
    return nullptr;
}
