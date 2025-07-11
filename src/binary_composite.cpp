/*
  Copyright (c) 2024-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include <memory>
#include <stdio.h>

#include "ispc.h"

#include <clang/Frontend/FrontendOptions.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/Path.h>

void printBinaryType() { printf("composite\n"); }

void initializeBinaryType(const char *ISPCExecutableAbsPath) {
    llvm::SmallString<128> includeDir(ISPCExecutableAbsPath);
    llvm::sys::path::remove_filename(includeDir);
    llvm::sys::path::remove_filename(includeDir);
    llvm::sys::path::append(includeDir, "include", "stdlib");
    ispc::g->includePath.push_back(std::string(includeDir.c_str()));
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
