// RUN: %{ispc} --target=host --nostdlib --nowrap -h %t.h %s 2>&1
// RUN: FileCheck %s --input-file=%t.h
// RUN: %{cc} -c -x c %t.h
// RUN: %{cc} -c -x c++ %t.h

// CHECK-COUNT-1: use(uint32_t test);

export void use(uniform uint test) {}
