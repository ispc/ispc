//; RUN: %{ispc} %s --nostdlib --target=sse4-i32x4,avx2-i32x4 --target-os=windows --emit-llvm-text -o %t.ll
//; RUN FileCheck %s %t.ll --check-prefix=DISPATCHER
//; RUN FileCheck %s %t_sse4.ll --check-prefix=TARGET
//; RUN FileCheck %s %t_avx2.ll --check-prefix=TARGET

// REQUIRES: X86_ENABLED && WINDOWS_ENABLED

// CHECK-DISPATCHER-COUNT-3: ; Function Attrs: uwtable
// CHECK-TARGET-COUNT-2: ; Function Attrs: {{.*}} uwtable
export uniform int foo() {
    return 2;
}