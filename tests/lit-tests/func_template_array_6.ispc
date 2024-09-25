// This tests checks multidimensional arrays inside nested templates
// RUN: %{ispc} %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK-LABEL: @test___
// CHECK: call {{.*}} @foo_level_1___vyiCuni64
// CHECK: @foo_level_1___vyiCuni64__
// CHECK: alloca [64 x [64 x [64 x i32]]]
// CHECK: call {{.*}} @foo_level_2___vyiCuni64
// CHECK: @foo_level_2___vyiCuni64
// CHECK: alloca [64 x [64 x [64 x i32]]]

template <typename T, int C>
noinline void foo_level_2(uniform float ret[], uniform int b[C][C][C]) {
    uniform int a[C][C][C];
    for (uniform int i = 0; i < C; ++i)
        for (uniform int j = 0; j < C; ++j)
            for (uniform int k = 0; k < C; ++k) {
                a[i][j][k] = 100*i+10*j+k;
                b[i][j][k] = 100*i+10*j-k;
            }

    ret[programIndex] = *(a[2][3] + 1);
}

template <typename T, int C>
noinline void foo_level_1(uniform float ret[]) {
    uniform int a[C][C][C];
    foo_level_2<T, C>(ret, a);
}

void test(uniform float ret[]) {
    foo_level_1<int, 64>(ret);
}
