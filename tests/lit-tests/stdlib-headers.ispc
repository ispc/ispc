// RUN: echo "" | %{ispc} -o - -E - | FileCheck %s

// CHECK: file core.isph
// CHECK: file stdlib.isph
