// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Malformed PlayStation triples and unsupported OSes must not select the glibc ABI.
// clang-format off
// RUN: not %{cxx} --target=x86_64-scei-ps -nostdlibinc -nostdinc++ -std=gnu++17 \
// RUN:   -I"%S/../.." -fsyntax-only "%S/../../builtins/builtins-c-cpu.cpp" 2>&1 | FileCheck %s
// RUN: not %{cxx} --target=x86_64-unknown-haiku -nostdlibinc -nostdinc++ -std=gnu++17 \
// RUN:   -I"%S/../.." -fsyntax-only "%S/../../builtins/builtins-c-cpu.cpp" 2>&1 | FileCheck %s
// REQUIRES: X86_ENABLED

// CHECK: error: "Unsupported target C library for CPU builtins"
// clang-format on
