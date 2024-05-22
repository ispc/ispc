// Test checking combination of different integer types used as template parameters.
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK-LABEL: define i64 @foo
// CHECK: call i64 @sum___Cuni8Cuni8Cuni16Cuni16Cuni32Cuni32Cuni64Cuni64___
// CHECK-LABEL: define linkonce_odr i64 @sum___Cuni8Cuni8Cuni16Cuni16Cuni32Cuni32Cuni64Cuni64___
// CHECK: ret i64 240

template<int8 x8, uint8 xu8, int16 x16, uint16 xu16, int x32, uint xu32, int64 x64, uint64 xu64>
noinline uniform uint64 sum() {
    return x8 + xu8 + x16 + xu16 + x32 + xu32 + x64 + xu64;
}

uniform uint64 foo() {
    return sum<8, 8, 16, 16, 32, 32, 64, 64>();
}