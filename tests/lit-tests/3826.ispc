// REQUIRES: X86_ENABLED

// RUN: %{ispc} %s -O0 --target=sse2-i32x4 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=SSE2
// RUN: %{ispc} %s -O0 --target=avx2-i32x8 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=AVX2
// RUN: %{ispc} %s -O0 --target=avx512skx-x16 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=SKX
// RUN: %{ispc} %s -O0 --target=sse2-i32x4,avx2-i32x8,avx512skx-x16 --nostdlib --nowrap --emit-llvm-text -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=DISPATCH

// SSE2: @owned_arr = {{.*}}constant [4 x i32] {{.*}}, align 16
// SSE2: @ext_arr = external {{.*}}constant [4 x i32]
// SSE2-NOT: @ext_arr = {{.*}}, align 16

// AVX2: @owned_arr = {{.*}}constant [4 x i32] {{.*}}, align 32
// AVX2: @ext_arr = external {{.*}}constant [4 x i32]
// AVX2-NOT: @ext_arr = {{.*}}, align 32

// SKX: @owned_arr = {{.*}}constant [4 x i32] {{.*}}, align 64
// SKX: @ext_arr = external {{.*}}constant [4 x i32]
// SKX-NOT: @ext_arr = {{.*}}, align 64

// DISPATCH: @owned_arr = {{.*}}constant [4 x i32] {{.*}}, align 64

const uniform int owned_arr[4] = {1, 2, 3, 4};
extern const uniform int ext_arr[4];

export void foo(uniform int * uniform out) {
    *out = owned_arr[0] + ext_arr[0];
}