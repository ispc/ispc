// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Multiple labels named "label" in function

void func(int x) {
 label:
    ;
 label:
     ;
}

// CHECK: Error: "goto" statements are only legal under "uniform" control flow

void func2(int x) {
    if (x < 0)
        goto label;

 label:
        ;
}

// CHECK: Error: "goto" statements are only legal under "uniform" control flow

void func3(int x) {
    cif (x < 0)
        goto label;

 label:
        ;
}

// CHECK: Error: "goto" statements are only legal under "uniform" control flow

void func4(int x) {
 label:

    for(int i =0 ;i<x;)
        goto label;
}

// CHECK: Error: No label named "label" found in current function.

void func5(int x) {
completelydifferentlabel:
    goto label;
}

// CHECK: Error: No label named "label" found in current function.

void func6(int x) {
libel:
babel:
lbel:
label2:
    goto label;
}