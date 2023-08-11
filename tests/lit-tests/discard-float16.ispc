// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - %s 2>&1 | FileCheck %s

// REQUIRES: ASAN_RUN

// CHECK-NOT: detected memory leaks

float16 2.0f16 = two_f16;
