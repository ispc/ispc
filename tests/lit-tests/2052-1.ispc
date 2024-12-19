// The tests checks that no extra load/store instructions is generated.
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --emit-asm  -o - | FileCheck %s
// REQUIRES: ARM_ENABLED

// CHECK-LABEL: VectorTransformVector___
// CHECK-DAG: ldp
// CHECK-DAG: ldr
// CHECK-DAG: fmul
// CHECK-DAG: fmla
// CHECK-DAG: ldp
// CHECK-DAG: fmla

// CHECK-NOT: add
// CHECK-NOT: ldp
// CHECK-NOT: ldr
struct FVector4 {
    float<4> V;
};

struct FMatrix {
    float<16> M;
};

inline uniform FVector4 operator*(const uniform FVector4 &A, const uniform FVector4 &B) {
    varying float S0, S1, Result;
    *((uniform FVector4 * uniform) & S0) = *((uniform FVector4 * uniform) & A);
    *((uniform FVector4 * uniform) & S1) = *((uniform FVector4 * uniform) & B);

    Result = S0 * S1;

    return *((uniform FVector4 * uniform) & Result);
}

inline uniform FVector4 SetVector4(const uniform float X, const uniform float Y, const uniform float Z,
                                   const uniform float W) {
    const uniform FVector4 Result = {{X, Y, Z, W}};
    return Result;
}

inline uniform FVector4 VectorMultiplyAdd(const uniform FVector4 &A, const uniform FVector4 &B, const uniform FVector4 &C) {
    return SetVector4(
        A.V[0] * B.V[0] + C.V[0],
        A.V[1] * B.V[1] + C.V[1],
        A.V[2] * B.V[2] + C.V[2],
        A.V[3] * B.V[3] + C.V[3]
    );
}

uniform FVector4 VectorTransformVector(const uniform FVector4 &Vec, const uniform FMatrix &M) {
    uniform FVector4 VTempX, VTempY, VTempZ, VTempW;

    // Splat x,y,z and w
    VTempX = SetVector4(Vec.V[0], Vec.V[0], Vec.V[0], Vec.V[0]);
    VTempY = SetVector4(Vec.V[1], Vec.V[1], Vec.V[1], Vec.V[1]);
    VTempZ = SetVector4(Vec.V[2], Vec.V[2], Vec.V[2], Vec.V[2]);
    VTempW = SetVector4(Vec.V[3], Vec.V[3], Vec.V[3], Vec.V[3]);

    // Mul by the matrix
    VTempX = VTempX * SetVector4(M.M[0], M.M[1], M.M[2], M.M[3]);
    VTempX = VectorMultiplyAdd(VTempY, SetVector4(M.M[4], M.M[5], M.M[6], M.M[7]), VTempX);
    VTempX = VectorMultiplyAdd(VTempZ, SetVector4(M.M[8], M.M[9], M.M[10], M.M[11]), VTempX);
    VTempX = VectorMultiplyAdd(VTempW, SetVector4(M.M[12], M.M[13], M.M[14], M.M[15]), VTempX);

    return VTempX;
}
