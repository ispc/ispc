// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - --emit-llvm-text %s 2>&1 | FileCheck %s

// CHECK-NOT: FATAL ERROR:
export uniform float test (uniform float a)
{
    return a *
}
