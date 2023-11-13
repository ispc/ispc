// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Unable to find any matching overload for call to function

void foo(int x);

void bar(int x) {
    foo();
}
