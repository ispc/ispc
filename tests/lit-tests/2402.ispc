// The test checks that ISPC doesn't produce SSE42 instructions like "pcmpgtq" on SSE41 platforms.

// RUN: %{ispc} %s --target=sse4.1-i8x16 --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4.1-i16x8 --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4.1-i32x4 --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4.1-i32x8 --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4-i8x16 --cpu=penryn --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4-i16x8 --cpu=penryn --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4-i32x4 --cpu=penryn --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4-i32x8 --cpu=penryn --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4-i8x16 --emit-asm --nostdlib -o - | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: %{ispc} %s --target=sse4-i16x8 --emit-asm --nostdlib -o - | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: %{ispc} %s --target=sse4-i32x4 --emit-asm --nostdlib -o - | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: %{ispc} %s --target=sse4-i32x8 --emit-asm --nostdlib -o - | FileCheck %s -check-prefix=CHECK_SSE4
// REQUIRES: X86_ENABLED

// CHECK_SSE4: pcmpgtq
bool foo(int64 a, int64 b) {
    return a>b;
}
