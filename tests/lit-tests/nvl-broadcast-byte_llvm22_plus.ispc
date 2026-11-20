// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_22_0+
// RUN: %{ispc} %s --target=avx10.2nvl-x8 --arch=x86-64 --target-os=linux -O3 --woff --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx10.2nvl-x16 --arch=x86-64 --target-os=linux -O3 --woff --emit-asm -o - | FileCheck %s

// CHECK-LABEL: broadcast4:
// CHECK: [[LOOP:\.LBB[0-9_]+]]:{{.*}}Inner Loop Header
// CHECK-NOT: vpbroadcastd
// CHECK-NOT: vpmovzxdq
// CHECK: j{{[a-z]+}} {{[[:space:]]*}}[[LOOP]]
export void broadcast4(uniform int8 src[], uniform int8 dst[], uniform int n) {
    foreach (i = 0 ... n) {
        dst[i] = src[i & ~3];
    }
}
