// This test checks function signature of externs with '__vectorcall' to ensure it has 'x86_vectorcallcc'.
// RUN: %{ispc}  %s --emit-llvm-text --vectorcall --target=host --target-os=windows -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// REQUIRES: WINDOWS_ENABLED

// CHECK:declare x86_vectorcallcc float @CopyFloat(float)

extern "C" uniform float __vectorcall CopyFloat(const uniform float Source);

export void CopyFloat_ispc(uniform float Result[], const uniform float Source[], const uniform int Iterations) {
    for(uniform int i = 0; i < Iterations; i++) {
        Result[i] = CopyFloat(Source[i]);
    }
}
