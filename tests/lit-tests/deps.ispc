// Test for -M/-MF/-MT/-MMM. Tests are not very sophisticated, but good enough to be sanity tests and check that no crashes happen.

// Single target
//; RUN: %{ispc} %s --nowrap -o deps.o --target=avx2-i32x8 -M  2>&1 | FileCheck %s -check-prefix=CHECK1
//; RUN: %{ispc} %s --nowrap -o deps.o --target=avx2-i32x8 -M -MT new_file.ispc 2>&1 | FileCheck %s -check-prefix=CHECK2
//; RUN: %{ispc} %s --nowrap -o deps.o --target=avx2-i32x8 -M -MF %t.d && FileCheck %s --input-file=%t.d -check-prefix=CHECK1
//; RUN: %{ispc} %s --nowrap -o deps.o --target=avx2-i32x8 -M -MF %t.d -MT new_file.ispc && FileCheck %s --input-file=%t.d -check-prefix=CHECK2
//; RUN: %{ispc} %s --nowrap -o deps.o --target=avx2-i32x8 -MMM %t.d && FileCheck %s --input-file=%t.d -check-prefix=CHECK3

// Multi target
//; RUN: %{ispc} %s --nowrap -o deps.o --target=avx2-i32x8,avx512skx-x16 -M | FileCheck %s -check-prefix=CHECK1
//; RUN: %{ispc} %s --nowrap -o deps.o --target=avx2-i32x8,avx512skx-x16 -M -MT new_file.ispc 2>&1 | FileCheck %s -check-prefix=CHECK2
//; RUN: %{ispc} %s --nowrap -o deps.o --target=sse4-i32x4,avx2-i32x8       -M -MF %t.d && FileCheck %s --input-file=%t.d -check-prefix=CHECK1
//; RUN: %{ispc} %s --nowrap -o deps.o --target=sse4-i32x4,avx2-i32x8       -M -MF %t.d -MT new_file.ispc && FileCheck %s --input-file=%t.d -check-prefix=CHECK2
//; RUN: %{ispc} %s --nowrap -o deps.o --target=sse2-i32x8,avx512knl-x16 -MMM %t.d && FileCheck %s --input-file=%t.d -check-prefix=CHECK3

//; REQUIRES: X86_ENABLED

//; CHECK1: deps.o: {{.*}}deps.ispc
//; CHECK2: new_file.ispc: {{.*}}deps.ispc
//; CHECK3: {{.*}}deps.ispc
uniform int i;
