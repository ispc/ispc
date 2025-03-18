// This test checks that we can properly instantiate a template function at
// the moment when we assign its address to a function pointer.

// RUN: %{ispc} --target=host --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s

float same_exp(float x);
uniform int same_exp(uniform int x);
uniform double same_exp(uniform double x);

typedef float (*FuncType)(float);
typedef uniform int (*FuncType2)(uniform int);

template <typename T> T same(T x) { return same_exp(x); }

// CHECK-LABEL: @foo___vyf
// CHECK-DAG: call <{{.*}}> @same_exp___vyf({{.*}})
float foo(float a) {
    uniform FuncType func_ptr = same;
    return func_ptr(a);
}

// CHECK-LABEL: @foo2___uni
// CHECK-DAG: call i32 @same_exp___uni({{.*}})
uniform int foo2(uniform int a) {
    uniform FuncType2 func_ptr = same;
    return func_ptr(a);
}

// CHECK-LABEL: @foo3___und
// CHECK-DAG:  call double @same_exp___und({{.*}})
typedef uniform double (FuncType3)(uniform double);
uniform double foo3(uniform double a) {
    uniform FuncType3 *uniform func_ptr = same;
    return func_ptr(a);
}

// CHECK-LABEL: @foo4___vyf
// CHECK-DAG:  [[RES:%.*]] = {{.*}}call <{{.*}}> @same_exp___vyf(<{{.*}}> %a, <{{.*}}> %__mask)
float foo4(float a) {
    FuncType func_ptr = same;
    return func_ptr(a);
}
