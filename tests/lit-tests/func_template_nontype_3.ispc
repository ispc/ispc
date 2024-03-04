// Different scenarios using noin-type parameters.
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s
// CHECK-LABEL: define <{{[0-9]*}} x i32> @bar1___
// CHECK: call i32 @foo1___uni15___
// CHECK-LABEL: define linkonce_odr i32 @foo1___uni15___
// CHECK: ret i32 15

// CHECK-LABEL: define <{{[0-9]*}} x i32> @bar2___un_3C_uni_3E_un_3C_uni_3E_
// CHECK: call void @foo2___vyi4864
// CHECK-LABEL: define linkonce_odr void @foo2___vyi4864
// CHECK: partial_inner_all_outer:
// CHECK-NEXT:  ret void

// CHECK-LABEL: define i64 @bar4___
// CHECK: call i64 @foo4___8___
// CHECK-LABEL: define linkonce_odr i64 @foo4___8___
// CHECK: ret i64 10

template<typename T, int N>
noinline T foo1() {
    T value = N;
    return value;
}

int bar1() {
    return foo1<uniform int, 15>();
}

#define COUNT 64
template<typename T, int incx, int incy, int N>
noinline void foo2(uniform T alpha, uniform T x[], uniform T y[]) {
    #pragma unroll(4)
    foreach(i = 0 ... N) {
        y[i * incy] += alpha * x[i * incx];
    }
}

int bar2(uniform int x[], uniform int y[]) {
    foo2<int, 4, 8, COUNT>(2, x, y);
    return x[programIndex] + y[programIndex];
}

template<int8 N>
noinline uniform uint64 foo4() {
    uniform uint64 t = 2 + N;
    return t;
}

uniform uint64 bar4(uniform int x[], uniform int y[]) {
    return foo4<8>();
}
