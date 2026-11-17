// Including the AMX header may leave all instruction templates uninstantiated on any target.
// RUN: %{ispc} %s --target=host --nowrap --werror --emit-llvm-text -o - | FileCheck %s

#include <amx.isph>

uniform int tile_id() { return amx_tmm5; }

// CHECK-LABEL: @tile_id___
// CHECK: ret i32 5
