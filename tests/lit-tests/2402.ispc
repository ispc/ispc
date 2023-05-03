// The test checks that ISPC doesn't produce SSE42 instructions like "pcmpgtq" on SSE41 platforms.
// It also checks that sse4.1 and sse4.2 targets can't be used together in multi-target compilation.

// RUN: %{ispc} %s --target=sse4.1-i8x16 --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4.1-i16x8 --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4.1-i32x4 --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4.1-i32x8 --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4.2-i8x16 --cpu=penryn --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4.2-i16x8 --cpu=penryn --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4.2-i32x4 --cpu=penryn --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4.2-i32x8 --cpu=penryn --emit-asm --nostdlib -o - | FileCheck %s --implicit-check-not "pcmpgtq"
// RUN: %{ispc} %s --target=sse4.2-i8x16 --emit-asm --nostdlib -o - | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: %{ispc} %s --target=sse4.2-i16x8 --emit-asm --nostdlib -o - | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: %{ispc} %s --target=sse4.2-i32x4 --emit-asm --nostdlib -o - | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: %{ispc} %s --target=sse4.2-i32x8 --emit-asm --nostdlib -o - | FileCheck %s -check-prefix=CHECK_SSE4

// RUN: not %{ispc} %s --nowrap --target=sse4.1-i32x8,sse4.2-i32x8 --nostdlib 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR
// RUN: not %{ispc} %s --nowrap --target=sse4.2-i32x8,sse4.1-i32x8 --nostdlib 2>&1 | FileCheck %s -check-prefix=CHECK_ERROR
// REQUIRES: X86_ENABLED

// CHECK_SSE4: pcmpgtq
// CHECK_ERROR: Error: Can't compile to multiple variants of sse4 target!
bool foo(int64 a, int64 b) {
    return a>b;
}
