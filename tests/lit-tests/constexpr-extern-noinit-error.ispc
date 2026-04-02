// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

extern constexpr uniform int E;

// CHECK: Error: Missing initializer for constexpr variable "E".
