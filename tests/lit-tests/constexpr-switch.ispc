// RUN: %{ispc} %s --target=host --nostdlib --emit-llvm-text -o - | FileCheck %s

constexpr uniform int pick(uniform int x) {
    switch (x) {
    case 1:
        return 4;
    case 2:
        return 7;
    default:
        return 3;
    }
}

uniform int buf[pick(2)];

// CHECK: @buf = {{.*}} [7 x i32]
