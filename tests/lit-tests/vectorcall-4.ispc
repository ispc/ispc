// This test checks function signature of externs to ensure it doesn't have 'x86_vectorcallcc' by default.
// RUN: %{ispc}  %s --emit-llvm-text --target=host -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// REQUIRES: WINDOWS_ENABLED

// CHECK:declare float @CopyFloat(float)

extern "C" uniform float CopyFloat(const uniform float Source);

export void CopyFloat_ispc(uniform float Result[], const uniform float Source[], const uniform int Iterations)
{
	for(uniform int i = 0; i < Iterations; i++)
	{
        Result[i] = CopyFloat(Source[i]);
	}
}
