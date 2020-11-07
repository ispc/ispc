// This test checks function signature of extern "C" to ensure it doesn't have 'x86_vectorcallcc' by default.
// This verifies that calls from ISPC to C are not broken.
// RUN: %{ispc}  %s --emit-llvm-text --vectorcall --target=host --target-os=windows -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// REQUIRES: WINDOWS_ENABLED

// CHECK:declare float @CopyFloat(float)

extern "C" uniform float CopyFloat(const uniform float Source);

export void CopyFloat_ispc(uniform float Result[], const uniform float Source[], const uniform int Iterations) {
    for(uniform int i = 0; i < Iterations; i++) {
        Result[i] = CopyFloat(Source[i]);
    }
}
