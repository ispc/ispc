// Non-type parameters in templates specializations
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s
// CHECK-LABEL: define {{.*}} @printNumber___Cuni42___
// CHECK-LABEL: define {{.*}} @foo___
// CHECK: call void @printNumber___Cuni42___
// CHECK: call void @printNumber___Cuni43___
// CHECK-LABEL: define linkonce_odr {{.*}} @printNumber___Cuni43___

template<int N>
noinline void printNumber() {
   print("Not specialized. N = %", N);
}

template<>
noinline void printNumber<42>() {
    print("Specialized for 42");
}

void foo() {
    printNumber<42>();
    printNumber<43>();
}