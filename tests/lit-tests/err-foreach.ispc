// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Can't assign to type "const varying int32" 

int foo() {
    foreach (i = 0 ... 10) {
        ++i;
    }
}

// CHECK: Error: Nested "foreach" statements are currently illegal

void func();

void foo2(uniform int a, uniform int b) {
    foreach (i = 0 ... a) {
        foreach (j = 0 ... b) {
            func();
        }
    }
}