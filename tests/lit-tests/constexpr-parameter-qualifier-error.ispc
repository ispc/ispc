// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

void bad(constexpr uniform int x) {}

// CHECK: Error: "constexpr" qualifier is illegal in function parameter declaration for parameter "x".
