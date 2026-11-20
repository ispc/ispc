// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_22_0+
// RUN: %{ispc} %s --target=avx10.2nvl-x16 --arch=x86-64 --target-os=linux -O3 --woff --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx10.2nvl-x16 --arch=x86-64 --target-os=linux -O3 --woff --opt=disable-gathers --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx10.2nvl-x16 --arch=x86-64 --target-os=linux -O3 --woff --emit-asm -o - | FileCheck %s --check-prefix=ASM
// RUN: %{ispc} %s --target=avx10.2nvl-x16 --arch=x86-64 --target-os=linux -O3 --woff --opt=disable-gathers --emit-asm -o - | FileCheck %s --check-prefix=ASM

// ASM-LABEL: gather_bytes:
// ASM: [[LOOP:\.LBB[0-9_]+]]:{{.*}}Inner Loop Header
// ASM-NOT: vpmovzxdq
// ASM: j{{[a-z]+}} {{[[:space:]]*}}[[LOOP]]

// The full gang extracts an i32 lane before widening the unsigned offset.
// CHECK: foreach_full_body:
// CHECK: [[INDEX:%[^ ]+]] = load <16 x i32>
// CHECK-NOT: zext <16 x i32>
// CHECK: [[LANE:%[^ ]+]] = extractelement <16 x i32> [[INDEX]], i64 0
// CHECK-NEXT: [[OFFSET:%[^ ]+]] = zext i32 [[LANE]] to i64
// CHECK-NEXT: {{.*}}getelementptr i8, ptr %src, i64 [[OFFSET]]

// The tail continues to use a masked index load and store and unsigned offsets.
// CHECK: [[MASKED:%[^ ]+]] = {{.*}}call <16 x i32> @llvm.masked.load
// CHECK: zext <16 x i32> [[MASKED]] to <16 x i64>
// CHECK: call void @llvm.masked.store
export void gather_bytes(uniform int8 src[], uniform uint32 idx[], uniform int8 dst[], uniform int n) {
    foreach (i = 0 ... n) {
        dst[i] = src[idx[i]];
    }
}
