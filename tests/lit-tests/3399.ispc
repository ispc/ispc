// This test checks that the code below folds to simple paddb instructions on SSE2/SSE4 targets.
// RUN: %{ispc} %s --target=sse2-i32x4 --emit-asm --nostdlib -o - | FileCheck %s
// RUN: %{ispc} %s --target=sse2-i32x8 --emit-asm --nostdlib -o - | FileCheck %s
// RUN: %{ispc} %s --target=sse4.2-i32x4 --emit-asm --nostdlib -o - | FileCheck %s
// RUN: %{ispc} %s --target=sse4.2-i32x8 --emit-asm --nostdlib -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: paddb
// CHECK-NEXT: retq
unmasked uniform int32<4> test_add(uniform int32<4> _A, uniform int32<4> _B) {
    uniform int32<4> Result;
    foreach (i = 0 ... 16) {
        ((uniform int8 *)&Result)[i] = ((uniform int8 *)&_A)[i] + ((uniform int8 *)&_B)[i];
    }
    return Result;
}
