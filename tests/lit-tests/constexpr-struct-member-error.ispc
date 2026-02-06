// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

struct Bad {
    constexpr uniform int m;
};

// CHECK: Error: "constexpr" qualifier is illegal in type names and struct member declarations.
