// This test checks whether ISPC generates correct C and C++ header regarding reference as parameter.

// RUN: %{ispc} --target=host --nostdlib --nowrap -h %t.h %s 2>&1
// RUN: FileCheck %s --input-file=%t.h
// RUN: %{cc} -c -x c %t.h
// RUN: %{cc} -c -x c++ %t.h

// CHECK: use(uint32_t &);
// CHECK: use(uint32_t *);

export void use(uniform uint &) {}
