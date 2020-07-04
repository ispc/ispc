// RUN: %{ispc} %s --target=host --emit-llvm-text -o - | FileCheck %s
// CHECK: ret i32 0
extern uniform int test() {
    for (uniform bool i = 0; i < (uniform bool)1; i += 1)
    {
        return 0;
    }
    return 1;
}
