// Copyright 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Declaring a function through a function typedef must report an error, not crash.
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --check-prefix=PLAIN --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DUNSIGNED --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --check-prefix=UNSIGNED --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"

typedef void G(int);

// PLAIN: Error: Declaring function "y" with a typedef'd function type is not supported.
// UNSIGNED: Error: "unsigned" qualifier is illegal with
// UNSIGNED: Error: Declaring function "y" with a typedef'd function type is not supported.
#ifdef UNSIGNED
unsigned G y;
#else
G y;
#endif
