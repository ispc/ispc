// Instantiation of templates with bool parameters using nested templates.
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK-LABEL: define {{.*}} @somecode_with_fixed_path1
// CHECK: call <[[WIDTH:.*]] x i32> @somecode___Cunbtrue

// CHECK-LABEL: define {{.*}} @somecode___Cunbtrue
// CHECK: ret <[[WIDTH:.*]] x i32> {{<i32 1|splat \(i32 1\)}}
// CHECK-NOT: ret <[[WIDTH:.*]] x i32> zeroinitializer

// CHECK-LABEL: define {{.*}} @somecode_with_fixed_path2
// CHECK: call <[[WIDTH:.*]] x i32> @somecode___Cunbfalse

// CHECK-LABEL: define {{.*}} @somecode___Cunbfalse
// CHECK: ret <[[WIDTH:.*]] x i32> zeroinitializer
// CHECK-NOT: ret <[[WIDTH:.*]] x i32> {{<i32 1|splat \(i32 1\)}}

template <int foo> bool true_call() {
    return true;
}

template <int foo> bool false_call() {
    return false;
}
template <bool foo> noinline unmasked int somecode() {
    if (foo) {
        return true_call<foo>();
    } else {
        return false_call<foo>();
    }
}

int somecode_with_fixed_path1() {
    return somecode<true>();
}

int somecode_with_fixed_path2() {
    return somecode<false>();
}
