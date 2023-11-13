// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't assign to type "const uniform int[[W:[0-9]+]]" on left-hand side of expression

int bar(){ 
    4 = 0;
}

