// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to dereference void pointer type 

int foo(void * uniform ptr) {
    print("%", ptr[0]);
}
