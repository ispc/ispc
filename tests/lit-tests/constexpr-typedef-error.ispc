// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

typedef constexpr uniform int CI;

// CHECK: Error: "constexpr" qualifier is illegal in typedef declarations.
