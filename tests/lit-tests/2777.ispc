// RUN: %{ispc} %s --target=avx2-i32x8 --emit-asm -o - | FileCheck %s

// REQUIRES: X86_ENABLED && LLVM_17_0+

// CHECK-NOT: vmovd
// CHECK-NOT: vpinsrd

struct FVector4f
{
	float V[4];
};

// Extra vmovd, vpinsrd after 2x vmaxps
inline uniform FVector4f VectorMax(const uniform FVector4f& V1, const uniform FVector4f& V2)
{
	varying float S0, S1, Result;
	*((uniform FVector4f *uniform)&S0) = *((uniform FVector4f *uniform)&V1);
	*((uniform FVector4f *uniform)&S1) = *((uniform FVector4f *uniform)&V2);

	Result = max(S0, S1);

	return *((uniform FVector4f *uniform)&Result);
}

export void foo(uniform float A[], uniform float B[])
{

	uniform FVector4f *uniform pA = (uniform FVector4f *uniform)A;
	uniform FVector4f *uniform pB = (uniform FVector4f *uniform)B;

        *pA = VectorMax(*pA, *pB);
}
