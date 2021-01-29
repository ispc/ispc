// This test checks if instruction hoisting happens with CFGSimplificationPass.
// This is verified by checking for all_on/some_on basic blocks in test_cond_uni.
// These blocks will be optimized away if hoisting happens.

//; RUN: %{ispc} %s --target=host --emit-llvm-text -O2 --nowrap -o - | FileCheck %s

void foo(uniform int &arg0, uniform int &arg1) {
    if (arg0 < arg1) {
        uniform int temp = arg1;
        arg1 = arg0;
        arg0 = temp;
    }
}

//; CHECK: define i32 @test_cond__uni___uniuni
//; CHECK-NOT: all_on
//; CHECK-NOT: some_on
uniform int test_cond__uni(uniform int arg0, uniform int arg1) {
        foo(arg0, arg1);
        if (arg0 > arg1)
            return 0;
        return 1;
}
