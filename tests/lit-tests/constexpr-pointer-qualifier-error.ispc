// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

uniform int g = 1;
uniform int * constexpr P = &g;

// CHECK: Error: {{.*}}unexpected 'constexpr'{{.*}}
