/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include <memory>
#include <stdio.h>

#include <clang/Frontend/FrontendOptions.h>
#include <llvm/Support/MemoryBufferRef.h>

void printBinaryType() { printf("composite\n"); }

void initializeBinaryType(const char *) {
    // do nothing
}

extern const char core_isph_cpp_header[];
extern int core_isph_cpp_length;
llvm::StringRef getCoreISPHRef() {
    llvm::StringRef ref(core_isph_cpp_header, core_isph_cpp_length);
    return ref;
}

extern const char stdlib_isph_cpp_header[];
extern int stdlib_isph_cpp_length;
llvm::StringRef getStdlibISPHRef() {
    llvm::StringRef ref(stdlib_isph_cpp_header, stdlib_isph_cpp_length);

    return ref;
}
