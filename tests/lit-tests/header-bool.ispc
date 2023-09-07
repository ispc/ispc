// This test checks whether ISPC generates correct C and C++ header regarding used bool function parameter

// RUN: %{ispc} --target=host --nostdlib --nowrap -h %t.h %s 2>&1
// RUN: %{cc} -c -x c++ %t.h
// RUN: %{cc} -c -x c %t.h
// RUN: %{cc} -c -x c %t.h -E -o - 2>&1 | FileCheck %s

// CHECK: _Bool

export void use(uniform bool x) {}
