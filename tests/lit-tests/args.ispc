// Test check error reporting during arguments validation.

//; RUN: not %{ispc} %s -o %t.o --nowrap --target=generic-1 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_1
//; RUN: not %{ispc} %s -o %t.s --nowrap --emit-asm --target=generic-1 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_2
//; RUN: not %{ispc} %s -o %t.cpp --nowrap --emit-c++ --target=sse4-i32x4 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR_3

//; CHECK_ERROR_1: Error: When using a "generic-*" compilation target, object file output can not be used.
//; CHECK_ERROR_2: Error: When using a "generic-*" compilation target, assembly output can not be used.
//; CHECK_ERROR_3: Error: When generating C++ output, one of the "generic-*" targets must be used.

uniform int i;
