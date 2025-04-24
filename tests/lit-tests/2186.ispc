// This test checks that there is no ICE on the following code.

// RUN: %{ispc} --target=host --nowrap --nostdlib --emit-llvm-text --debug-phase=210:210 %s -o - | FileCheck %s

// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process
typedef float<4> vec4;

struct MyStruct {
    float* data;
};

// CHECK-LABEL: @GetData1
// CHECK-COUNT-4: __pseudo_masked_store_float
unmasked vec4 GetData1(MyStruct& from, int index) {
    return ((vec4*)from.data)[index];
}
// CHECK-LABEL: @GetData2
// CHECK-COUNT-4: __pseudo_masked_store_float
unmasked vec4 GetData2(MyStruct& from, int index) {
    return ((vec4*)from.data)->x;
}
