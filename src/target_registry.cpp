/*
  Copyright (c) 2019-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file target_registry.h
    @brief Registry to handle bitcode libraries.
*/

#include "target_registry.h"
#include "util.h"

#include <numeric>
#include <set>
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

    Triple(ISPCTarget target, TargetOS os, Arch arch) : m_target(target), m_os(os), m_arch(arch) {};

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
    m_dispatch = nullptr;
    m_dispatch_macos = nullptr;
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
            // PS5 is an alias to PS4 in terms of target files. All the tuning is done through CPU flags.
            if (lib->getOS() == TargetOS::ps4) {
                m_builtins[Triple(lib->getISPCTarget(), TargetOS::ps5, lib->getArch()).encode()] = lib;
                m_supported_oses[(int)TargetOS::ps5] = true;
            }
            break;
        case BitcodeLib::BitcodeLibType::ISPC_target:
            m_targets[Triple(lib->getISPCTarget(), lib->getOS(), lib->getArch()).encode()] = lib;
            // "custom_linux" target is regular "linux" target for ARM with a few tweaks.
            // So, create it as an alias.
            if (lib->getOS() == TargetOS::linux && (lib->getArch() == Arch::arm || lib->getArch() == Arch::aarch64)) {
                m_targets[Triple(lib->getISPCTarget(), TargetOS::custom_linux, lib->getArch()).encode()] = lib;
            }
            // PS5 is an alias to PS4 in terms of target files. All the tuning is done through CPU flags.
            if (lib->getOS() == TargetOS::ps4) {
                m_targets[Triple(lib->getISPCTarget(), TargetOS::ps5, lib->getArch()).encode()] = lib;
            }
            break;
        case BitcodeLib::BitcodeLibType::Stdlib:
            m_stdlibs[Triple(lib->getISPCTarget(), lib->getOS(), lib->getArch()).encode()] = lib;
            // "custom_linux" target is regular "linux" target for ARM with a few tweaks.
            if (lib->getOS() == TargetOS::linux && (lib->getArch() == Arch::arm || lib->getArch() == Arch::aarch64)) {
                m_stdlibs[Triple(lib->getISPCTarget(), TargetOS::custom_linux, lib->getArch()).encode()] = lib;
            }
            // PS5 is an alias to PS4 in terms of target files. All the tuning is done through CPU flags.
            if (lib->getOS() == TargetOS::ps4) {
                m_stdlibs[Triple(lib->getISPCTarget(), TargetOS::ps5, lib->getArch()).encode()] = lib;
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
static const BitcodeLib *lGetTargetLib(const std::map<uint32_t, const BitcodeLib *> &libs, ISPCTarget target,
                                       TargetOS os, Arch arch) {
    // TODO: validate parameters not to be errors or forbidden values.

    // This is an alias. It might be a good idea generalize this.
    if (target == ISPCTarget::avx1_i32x4) {
        target = ISPCTarget::sse4_i32x4;
    }

    // sse41 is an alias for sse4
    switch (target) {
    case ISPCTarget::sse41_i8x16:
        target = ISPCTarget::sse4_i8x16;
        break;
    case ISPCTarget::sse41_i16x8:
        target = ISPCTarget::sse4_i16x8;
        break;
    case ISPCTarget::sse41_i32x4:
        target = ISPCTarget::sse4_i32x4;
        break;
    case ISPCTarget::sse41_i32x8:
        target = ISPCTarget::sse4_i32x8;
        break;
    default:
        // Fall through
        ;
    }

    // There's no Mac that supports SPR, so the decision is not support these targets when targeting macOS.
    // If these targets are linked in, then we still can use them for cross compilation, for example for Linux.
    if (os == TargetOS::macos && (target == ISPCTarget::avx512spr_x4 || target == ISPCTarget::avx512spr_x8 ||
                                  target == ISPCTarget::avx512spr_x16 || target == ISPCTarget::avx512spr_x32 ||
                                  target == ISPCTarget::avx512spr_x64)) {
        return nullptr;
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
    case TargetOS::ps5:
        os = TargetOS::linux;
        break;
    case TargetOS::error:
        UNREACHABLE();
    }

    auto result = libs.find(Triple(target, os, arch).encode());
    if (result != libs.end()) {
        return result->second;
    }
    return nullptr;
}

const BitcodeLib *TargetLibRegistry::getISPCTargetLib(ISPCTarget target, TargetOS os, Arch arch) const {
    return lGetTargetLib(m_targets, target, os, arch);
}

const BitcodeLib *TargetLibRegistry::getISPCStdLib(ISPCTarget target, TargetOS os, Arch arch) const {
    return lGetTargetLib(m_stdlibs, target, os, arch);
}

std::vector<std::string> TargetLibRegistry::checkBitcodeLibs() const {
    std::vector<std::string> missedFiles = {};
    if (!g->isSlimBinary) {
        return missedFiles;
    }
    for (ISPCTarget target = ISPCTarget::sse2_i32x4; target < ISPCTarget::error; target++) {
        for (TargetOS os = TargetOS::windows; os < TargetOS::error; os++) {
            for (Arch arch = Arch::none; arch < Arch::error; arch++) {
                if (isSupported(target, os, arch)) {
                    const BitcodeLib *clib = getBuiltinsCLib(os, arch);
                    const BitcodeLib *tlib = getISPCTargetLib(target, os, arch);
                    const BitcodeLib *slib = getISPCStdLib(target, os, arch);
                    if (!clib->fileExists()) {
                        missedFiles.push_back(clib->getFilename());
                    }
                    if (!tlib->fileExists()) {
                        missedFiles.push_back(tlib->getFilename());
                    }
                    if (!slib->fileExists()) {
                        missedFiles.push_back(slib->getFilename());
                    }
                }
            }
        }
    }
    return missedFiles;
}

// Print user-friendly message about supported targets
void TargetLibRegistry::printSupportMatrix() const {
    // Vector of rows, which are vectors of cells.
    std::vector<std::vector<std::string>> table;

    // OS names row
    std::vector<std::string> os_names;
    os_names.push_back("");
    for (TargetOS os = TargetOS::windows; os < TargetOS::error; os++) {
        os_names.push_back(OSToString(os));
    }
    table.push_back(os_names);

    // Fill in the name, one target per the row.
    for (ISPCTarget target = ISPCTarget::sse2_i32x4; target < ISPCTarget::error; target++) {
        std::vector<std::string> row;
        row.push_back(ISPCTargetToString(target));
        std::vector<std::string> arch_list_target;
        // Fill in cell: list of arches for the target/os.
        for (TargetOS os = TargetOS::windows; os < TargetOS::error; os++) {
            std::string arch_list_os;
            for (Arch arch = Arch::none; arch < Arch::error; arch++) {
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
    for (Arch arch = Arch::none; arch < Arch::error; arch++) {
        for (ISPCTarget target = ISPCTarget::sse2_i32x4; target < ISPCTarget::error; target++) {
            for (TargetOS os = TargetOS::windows; os < TargetOS::error; os++) {
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
    std::set<std::string> targetSet;
    for (Arch arch = Arch::none; arch < Arch::error; arch++) {
        for (ISPCTarget target = ISPCTarget::sse2_i32x4; target < ISPCTarget::error; target++) {
            for (TargetOS os = TargetOS::windows; os < TargetOS::error; os++) {
                if (isSupported(target, os, arch)) {
                    targetSet.insert(ISPCTargetToString(target));
                    break;
                }
            }
        }
    }

    std::string targets;
    for (const auto &target : targetSet) {
        if (!targets.empty()) {
            targets += ", ";
        }
        targets += target;
    }

    return targets;
}

std::string TargetLibRegistry::getSupportedOSes() {
    // We use pre-computed bitset, as this function is perfomance critical - it's used
    // during arguments parsing.
    std::string oses;
    for (TargetOS os = TargetOS::windows; os < TargetOS::error; os++) {
        if (m_supported_oses[static_cast<int>(os)]) {
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
