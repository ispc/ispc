// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - %s 2>&1 | FileCheck %s

// REQUIRES: ASAN_RUN

// CHECK-NOT: detected memory leaks

struct SNAME { int x; };
int x = SNAME;
