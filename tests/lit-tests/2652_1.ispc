// RUN: %{ispc} --target=host --nostdlib --nowrap -h %t.h %s 2>&1
// RUN: FileCheck %s --input-file=%t.h
// RUN: %{cc} -c -x c %t.h
// RUN: %{cc} -c -x c++ %t.h

// CHECK-COUNT-1: uint32_t my_data;

struct Test {
  uint my_data;
};

export void use(uniform Test &test) {}
