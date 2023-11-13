// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "goto" statements are only legal under "uniform" control flow

void func(int x) {
    if (x < 0)
        goto label;

 label:
        ;
}


