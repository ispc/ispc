/*
  Copyright (c) 2013-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

// This is the source code of __get_system_isa and __set_system_isa functions
// for dispatch built-in module.
//
// This file is compiled with clang during ISPC build in the following way:
// - clang dispatch.c -O2 -emit-llvm -S -c -DREGULAR -o isa_dispatch.ll
// - clang dispatch.c -O2 -emit-llvm -S -c -DMACOS   -o isa_dispatch-macos.ll
//
// MACOS version: the key difference is absence of OS support check for AVX512
// - see issue #1854 for more details. Also it does not support ISAs newer than
// SKX, as no such Macs exist.

// Require one of the macros to be defined to make sure that it's not
// misspelled on the command line.
#if !defined(REGULAR) && !defined(MACOS)
#error "Either REGULAR or MACOS macro need to defined"
#endif

// We include isa.h here as the simplest way to get the get_x86_isa
// function in this module. Note, that this module is translated to LLVM IR. We
// may also translate isa.h to LLVM IR separately and link it with this module
// via llvm-link. However, this approach requires llvm-link as a dependency
// during ISPC build time.
#include "isa.h"

static int __system_best_isa = -1;

// For function definitions, we need to use static keyword. This is because
// because users can compile several translation units in multi-target mode and
// link them together. Putting static let's us avoid the linker reporting
// multiple definitions of the same function error.

// We don't apply static for __terminate_now and __get_system_isa functions
// here because we need to preserve them until they are linked with the main
// user code during user code compilation. Then, we will assign them the
// internal linkage in builtins.cpp::LinkInDispatcher function.
void __terminate_now() {
    // Terminate execution using x86 UD2 instruction. UD2 raises an invalid
    // opcode exception, ensuring program termination.
    __asm__ __volatile__("ud2");
}

// __get_system_isa should return a value corresponding to one of the
// Target::ISA enumerant values that gives the most capable ISA that the
// current system can run.
static int __get_system_isa() {
    enum ISA isa = get_x86_isa();

    if (isa == INVALID || isa == KNL_AVX512) {
        __terminate_now();
        return -1;
    }

    if (isa == AVX11) {
        return AVX;
    }

    return isa;
}

void __set_system_isa() {
    if (__system_best_isa == -1) {
        __system_best_isa = __get_system_isa();
    }
}
