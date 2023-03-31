/*
  Copyright (c) 2019-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file target_registry.h
    @brief Registry to handle bitcode libraries.
*/

#pragma once

#include "bitcode_lib.h"

#include <bitset>
#include <cstdint>
#include <map>
#include <vector>

namespace ispc {
// Some background information useful for understanding how this works.
// - static variables with global or class scope and without constructor are
//   initialized at load time.
// - static variables with global or class scope, which require code execution
//   (i.e. constructor), are initialized after load time, but before the main()
//   is started.  The order of initialization is defined only inside single
//   translation unit.  Between translation units the order is not defined by
//   the language  (defined by linker implementation).  Initialization guaranteed
//   to happen in a single thread.
// - function scope static variables are initialized on the first use.

// Initialization of TargetLibRegistry happens in two stages.
// On the first stage libraries are registered by executing BitcodeLib constructors
// of their static instances.  The pointers to the libs are stored in the vector
// (TargetLibRegistry::libs).
// On the second stage, an instance of TargetLibRegistry is constructed after the
// main() starts.  During TargetLibRegistry construction TargetLibRegistry::libs
// vector is used to populate the class data and store information that is
// convenient for further access.  TargetLibRegistry::libs is nullified at this point.

class TargetLibRegistry {
    static std::vector<const BitcodeLib *> *libs;
    TargetLibRegistry();

    // Dispatch
    // Currently only one dispatch module is supported: for x86 CPUs.
    // It's OS / arch agnostic, except for macOS, see issue 1854 for more details.
    // TODO: do we need separate dispatch module for Windows and for Unix?
    const BitcodeLib *m_dispatch;
    const BitcodeLib *m_dispatch_macos;

    // Builtins-c
    // OS x Arch
    std::map<uint32_t, const BitcodeLib *> m_builtins;

    // ISPC targets
    // Target x OS (Win/Unix) x Arch [32/64/none]
    std::map<uint32_t, const BitcodeLib *> m_targets;

    // Bitset with supported OSes
    std::bitset<(int)TargetOS::error> m_supported_oses;

  public:
    static void RegisterTarget(const BitcodeLib *lib);
    static TargetLibRegistry *getTargetLibRegistry();

    // Return dispatch module if available, otherwise nullptr.
    const BitcodeLib *getDispatchLib(const TargetOS os) const;

    // Return builtins-c module if available, otherwise nullptr.
    const BitcodeLib *getBuiltinsCLib(TargetOS os, Arch arch) const;

    // Return target module if available, otherwise nullptr.
    const BitcodeLib *getISPCTargetLib(ISPCTarget target, TargetOS os, Arch arch) const;

    // Print user-friendly message about supported targets
    void printSupportMatrix() const;

    std::string getSupportedArchs();
    std::string getSupportedTargets();
    std::string getSupportedOSes();

    bool isSupported(ISPCTarget target, TargetOS os, Arch arch) const;
};
} // namespace ispc
