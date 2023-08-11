// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - %s 2>&1 | FileCheck %s

// REQUIRES: ASAN_RUN

// CHECK-NOT: detected memory leaks


const int r = 7;

r = 1;
