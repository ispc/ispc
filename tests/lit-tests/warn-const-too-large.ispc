// RUN: %{ispc} --target=host --nowrap --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s

// CHECK: Warning: Binary expression with type "const uniform int8" can't represent value.

int8 v = (int8)64 * (int8)32;

// CHECK: Warning: Binary expression with type "const uniform int32" can't represent value.
uniform int32 v32 = 1073741824 * 2;

// CHECK: Warning: Binary expression with type "const uniform int64" can't represent value.
uniform int64 v64 = 9223372036854775807ll + 1ll;
