//; RUN: %{ispc} %s -o %t.o --nowrap --target=host 2>&1 | FileCheck %s
//; CHECK: Warning: Exported function parameter "arg1" points to varying
export void foo1(varying int * uniform arg1){
    return;
}

//; CHECK: Warning: Exported function parameter "arg2" points to varying
export void foo2(varying int arg2[10]){
    return;
}

struct foo_struct3 {
    int i;
};
//; CHECK: Warning: Exported function parameter "arg3" points to varying
export void foo3(varying foo_struct3 * uniform arg3){
    return;
}

struct foo_struct4 {
    uniform int i;
};
//; CHECK-NOT: Warning:
export void foo4(varying foo_struct4 * uniform arg4){
    return;
}
