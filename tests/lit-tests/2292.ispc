// This test checks that generated code does not try to divide by zero.

// RUN: %{ispc} --pic --target=host %s -o %t.o
// RUN: %{cc} %t.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK-NOT: Floating point exception
// CHECK: [1,((

export void test(){
  varying int x = 0;
  varying int y = 0;
  foreach_active(i){
    if(i==0){
      x = 1;
      y = 1;
      print("%\n", x/y);
    }
  }
}

extern "C" uniform int main() {
    test();
}
