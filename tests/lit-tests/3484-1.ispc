// This test ensures that the compiler doesn't crash when trying to initialize
// a uniform vector with varying values, but instead produces an appropriate error message.
// Previously this would cause an LLVM assertion failure in FixedVectorType::get().

// RUN: not %{ispc} %s --target=host --nowrap 2>&1 | FileCheck %s

// CHECK: Error: Can't convert from type "varying float" to type "uniform float" for initializer list

static varying float load(const uniform float *values) { return 0.0f; }

export void func(uniform float values[][programCount]) {
    uniform float<3> col_xyz = {load(&values[0][0]),
                                load(&values[0][0]),
                                load(&values[0][0])};
}