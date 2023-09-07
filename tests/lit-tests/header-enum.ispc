// This test checks whether ISPC generates correct C and C++ header regarding used enum function parameter.

// RUN: %{ispc} --target=host --nostdlib --nowrap -h %t.h %s 2>&1
// RUN: FileCheck %s --input-file=%t.h
// RUN: %{cc} -c -x c %t.h
// RUN: %{cc} -c -x c++ %t.h

// CHECK: use(enum Color);

enum Color { RED, GREEN, BLUE };

export void use(uniform Color) {}

