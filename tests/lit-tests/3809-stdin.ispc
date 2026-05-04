//; RUN: cat %s | %{ispc} - --target=host --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

#define MAXI 5

// CHECK-LABEL: @test___
export void test(uniform int output[]) {
    // CHECK: store
    foreach (i = 0...MAXI) {
        output[i] = i;
    }
}
