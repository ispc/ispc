// RUN: %{ispc} %s --target=host -O0 --emit-llvm-text --nowrap --nostdlib -o - | FileCheck %s

// Test global variable declaration followed by definition
extern uniform int globalVar1;
uniform int globalVar1 = 42;

// Test multiple declarations of the same extern variable
extern uniform float globalVar2;
extern uniform float globalVar2;

// Test global variable with initializer
static uniform int staticVar = 123;

// Test const global variable
const uniform int constVar = 456;

// CHECK: @globalVar1 = global i32 42
// CHECK: @globalVar2 = external global float
// CHECK: @staticVar = internal global i32 123
// CHECK: @constVar = constant i32 456

void test() {
    // Use the variables to ensure they're not optimized away
    uniform int sum = globalVar1 + staticVar + constVar;
    globalVar2 = 3.14f;
}