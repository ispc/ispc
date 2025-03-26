// RUN: %{ispc} --target=host --nowrap --debug-phase=10:10 %s -o %t 2>&1 | FileCheck %s

struct S { int16 f[3]; };
struct S4 { int16 f[3]; } __attribute__((aligned(4)));
struct S8 { int16 f[3]; } __attribute__((aligned(8)));
struct S16 { int16 f[3]; } __attribute__((aligned(16)));
struct G { int8 x; } __attribute__((aligned(2)));

void foo() {
// CHECK: @static.{{.*}}.u0 = internal global %v{{.*}}_uniform_S zeroinitializer
    static uniform struct S u0;

// CHECK: @static.{{.*}}.u64 = internal global %v{{.*}}_uniform_S zeroinitializer, align 64
    __attribute__((aligned(64))) static uniform struct S u64;

// CHECK: @static.{{.*}}.u4 = internal global %v{{.*}}_uniform_S4 zeroinitializer, align 4
    static uniform struct S4 u4;

// CHECK: @static.{{.*}}.u8 = internal global %v{{.*}}_uniform_S8 zeroinitializer, align 8
    static uniform struct S8 u8;

// CHECK: @static.{{.*}}.u16 = internal global %v{{.*}}_uniform_S16 zeroinitializer, align 16
    static uniform struct S16 u16;

// CHECK: @static.{{.*}}.uc = internal global %v{{.*}}_uniform_G zeroinitializer, align 2
    static uniform struct G uc;

    u0.f[0] = 1;
    u64.f[0] = 1;
    u4.f[0] = 1;
    u8.f[0] = 1;
    u16.f[0] = 1;
    uc.x = 1;
}
