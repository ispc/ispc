// RUN: %{ispc} --target=host --no-discard-value-names --nowrap --debug-phase=10:10 %s -o %t 2>&1 | FileCheck %s

struct S { int16 f[3]; };
struct S4 { int16 f[3]; } __attribute__((aligned(4)));
struct S8 { int16 f[3]; } __attribute__((aligned(8)));
struct S16 { int16 f[3]; } __attribute__((aligned(16)));
struct G { int8 x; } __attribute__((aligned(2)));

// CHECK-LABEL: @foo___
void foo() {
// CHECK-DAG: %u0 = alloca %v{{.*}}_uniform_S, align 8
    uniform struct S u0;

// CHECK-DAG: %u64 = alloca %v{{.*}}_uniform_S, align 64
    __attribute__((aligned(64))) uniform struct S u64;

// CHECK-DAG: %u4 = alloca %v{{.*}}_uniform_S4, align 4
    uniform struct S4 u4;

// CHECK-DAG: %u8 = alloca %v{{.*}}_uniform_S8, align 8
    uniform struct S8 u8;

// CHECK-DAG: %u16 = alloca %v{{.*}}_uniform_S16, align 16
    uniform struct S16 u16;

// CHECK-DAG: %uc = alloca %v{{.*}}_uniform_G, align 2
    uniform struct G uc;

    u0.f[0] = 1;
    u64.f[0] = 1;
    u4.f[0] = 1;
    u8.f[0] = 1;
    u16.f[0] = 1;
    uc.x = 1;
}
