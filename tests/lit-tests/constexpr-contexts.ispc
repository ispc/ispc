// RUN: %{ispc} %s --target=host --nostdlib --emit-llvm-text -o - | FileCheck %s

constexpr uniform int width() { return 4; }
constexpr uniform int four() { return 4; }

enum E {
    EA = width(),
    EB = four() + 2
};

uniform int<width()> gv;
uniform int arr[EB];

uniform int use_case_values(uniform int x) {
    switch (x) {
    case four():
        return 11;
    case EB:
        return 22;
    default:
        return 33;
    }
}

// CHECK-DAG: @gv = {{.*}} <4 x i32> zeroinitializer
// CHECK-DAG: @arr = {{.*}} [6 x i32] zeroinitializer
// CHECK-LABEL: @use_case_values
// CHECK: icmp eq i32 %{{.*}}, 6
// CHECK: icmp eq i32 %{{.*}}, 4
