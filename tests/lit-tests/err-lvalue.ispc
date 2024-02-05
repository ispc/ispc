// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Can't assign to type "const uniform int[[W:[0-9]+]]" on left-hand side of expression

int bar2(){ 
    4 = 0;
}

// CHECK: Error: Can't assign to type "const uniform int[[W:[0-9]+]]" on left-hand side of expression

int bar3(){ 
    int x;
    4 = x;
}

// CHECK: Error: Illegal to take address of non-lvalue or function

void func() {
    int *ptr = &(1+1);
}

// CHECK: Error: Left hand side of assignment expression can't be assigned to

int foo() {return 2;}

int bar()
{
    foo() = 2;
}