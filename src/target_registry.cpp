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

/** @file target_registry.h
    @brief Registry to handle bitcode libraries.
*/

#include "target_registry.h"
#include "util.h"

#include <numeric>
#include <string>
#include <vector>

using namespace ispc;

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
    m_dispatch = NULL;
    m_dispatch_macos = NULL;
    for (auto lib : *libs) {
        switch (lib->getType()) {
        case BitcodeLib::BitcodeLibType::Dispatch:
            if (lib->getOS() == TargetOS::macos) {
                m_dispatch_macos = lib;
            } else {
                m_dispatch = lib;
            }
            break;
        case BitcodeLib::BitcodeLibType::Builtins_c:
            m_builtins[Triple(lib->getISPCTarget(), lib->getOS(), lib->getArch()).encode()] = lib;
            m_supported_oses[(int)lib->getOS()] = true;
            // "custom_linux" target is regular "linux" target for ARM with a few tweaks.
            // So, create it as an alias.
            if (lib->getOS() == TargetOS::linux && (lib->getArch() == Arch::arm || lib->getArch() == Arch::aarch64)) {
                m_builtins[Triple(lib->getISPCTarget(), TargetOS::custom_linux, lib->getArch()).encode()] = lib;
                m_supported_oses[(int)TargetOS::custom_linux] = true;
            }
            break;
        case BitcodeLib::BitcodeLibType::ISPC_target:
            m_targets[Triple(lib->getISPCTarget(), lib->getOS(), lib->getArch()).encode()] = lib;
            // "custom_linux" target is regular "linux" target for ARM with a few tweaks.
            // So, create it as an alias.
            if (lib->getOS() == TargetOS::linux && (lib->getArch() == Arch::arm || lib->getArch() == Arch::aarch64)) {
                m_targets[Triple(lib->getISPCTarget(), TargetOS::custom_linux, lib->getArch()).encode()] = lib;
            }
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

const BitcodeLib *TargetLibRegistry::getDispatchLib(const TargetOS os) const {
    return (os == TargetOS::macos) ? m_dispatch_macos : m_dispatch;
}

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

    // Canonicalize OS, as for the target we only differentiate between Windows, Unix, and Web (WASM target).
    switch (os) {
    case TargetOS::windows:
    case TargetOS::web:
        // Keep these values.
        break;
    case TargetOS::linux:
    case TargetOS::custom_linux:
    case TargetOS::freebsd:
    case TargetOS::macos:
    case TargetOS::android:
    case TargetOS::ios:
    case TargetOS::ps4:
        os = TargetOS::linux;
        break;
    case TargetOS::error:
        UNREACHABLE();
    }

    auto result = m_targets.find(Triple(target, os, arch).encode());
    if (result != m_targets.end()) {
        return result->second;
    }
    return nullptr;
}

// Print user-friendly message about supported targets
void TargetLibRegistry::printSupportMatrix() const {
    // Vector of rows, which are vectors of cells.
    std::vector<std::vector<std::string>> table;

    // OS names row
    std::vector<std::string> os_names;
    os_names.push_back("");
    for (int j = (int)TargetOS::windows; j < (int)TargetOS::error; j++) {
        os_names.push_back(OSToString((TargetOS)j));
    }
    table.push_back(os_names);

    // Fill in the name, one target per the row.
    for (int i = (int)ISPCTarget::sse2_i32x4; i < (int)ISPCTarget::error; i++) {
        std::vector<std::string> row;
        ISPCTarget target = (ISPCTarget)i;
        row.push_back(ISPCTargetToString(target));
        std::vector<std::string> arch_list_target;
        // Fill in cell: list of arches for the target/os.
        for (int j = (int)TargetOS::windows; j < (int)TargetOS::error; j++) {
            std::string arch_list_os;
            TargetOS os = (TargetOS)j;
            for (int k = (int)Arch::none; k < (int)Arch::error; k++) {
                Arch arch = (Arch)k;
                if (isSupported(target, os, arch)) {
                    if (!arch_list_os.empty()) {
                        arch_list_os += ", ";
                    }
                    arch_list_os += ArchToString(arch);
                }
            }
            arch_list_target.push_back(arch_list_os);
            row.push_back(arch_list_os);
        }
        table.push_back(row);
    }

    // Collect maximum sizes for all columns
    std::vector<int> column_sizes(table[0].size(), 7);
    for (auto &row : table) {
        for (int i = 0; i < row.size(); i++) {
            column_sizes[i] = column_sizes[i] > row[i].size() ? column_sizes[i] : row[i].size();
        }
    }
    int width = std::accumulate(column_sizes.begin(), column_sizes.end(), 0) + (column_sizes.size() - 1) * 3;

    // Print the table
    for (int i = 0; i < table.size(); i++) {
        auto row = table[i];
        for (int j = 0; j < row.size(); j++) {
            auto align = std::string(column_sizes[j] - row[j].size(), ' ');
            printf("%s%s", row[j].c_str(), align.c_str());
            if (j + 1 != row.size()) {
                printf(" | ");
            }
        }
        printf("\n");
        if (i == 0) {
            auto line = std::string(width, '-');
            printf("%s\n", line.c_str());
        }
    }
}

std::string TargetLibRegistry::getSupportedArchs() {
    std::string archs;
    for (int k = (int)Arch::none; k < (int)Arch::error; k++) {
        Arch arch = (Arch)k;
        for (int i = (int)ISPCTarget::sse2_i32x4; i < (int)ISPCTarget::error; i++) {
            ISPCTarget target = (ISPCTarget)i;
            for (int j = (int)TargetOS::windows; j < (int)TargetOS::error; j++) {
                TargetOS os = (TargetOS)j;

                if (isSupported(target, os, arch)) {
                    if (!archs.empty()) {
                        archs += ", ";
                    }
                    archs += ArchToString(arch);
                    goto next_arch;
                }
            }
        }
    next_arch:;
    }

    return archs;
}

std::string TargetLibRegistry::getSupportedTargets() {
    std::string targets;
    for (int i = (int)ISPCTarget::sse2_i32x4; i < (int)ISPCTarget::error; i++) {
        ISPCTarget target = (ISPCTarget)i;
        for (int j = (int)TargetOS::windows; j < (int)TargetOS::error; j++) {
            TargetOS os = (TargetOS)j;
            for (int k = (int)Arch::none; k < (int)Arch::error; k++) {
                Arch arch = (Arch)k;
                if (isSupported(target, os, arch)) {
                    if (!targets.empty()) {
                        targets += ", ";
                    }
                    targets += ISPCTargetToString(target);
                    goto next_target;
                }
            }
        }
    next_target:;
    }

    return targets;
}

std::string TargetLibRegistry::getSupportedOSes() {
    // We use pre-computed bitset, as this function is perfomance critical - it's used
    // during arguments parsing.
    std::string oses;
    for (int j = (int)TargetOS::windows; j < (int)TargetOS::error; j++) {
        TargetOS os = (TargetOS)j;
        if (m_supported_oses[j]) {
            if (!oses.empty()) {
                oses += ", ";
            }
            oses += OSToLowerString(os);
        }
    }

    return oses;
}

bool TargetLibRegistry::isSupported(ISPCTarget target, TargetOS os, Arch arch) const {
    auto clib = getBuiltinsCLib(os, arch);
    if (clib) {
        auto lib = getISPCTargetLib(target, os, arch);
        if (lib) {
            return true;
        }
    }
    return false;
}
