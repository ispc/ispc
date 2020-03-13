struct FVector {
    float V[3];
};

inline uniform FVector Mul1(const uniform FVector &A, const uniform FVector &B) {
    uniform FVector Result;

    foreach (i = 0 ... 3) { Result.V[i] = A.V[i] * B.V[i]; }

    return Result;
}

inline uniform FVector Mul2(const uniform FVector &A, const uniform FVector &B) {
    uniform FVector Result;

    for (uniform int i = 0; i < 3; i++) {
        Result.V[i] = A.V[i] * B.V[i];
    }

    return Result;
}

inline uniform FVector Mul3(const uniform FVector &A, const uniform FVector &B) {
    uniform float<3> S0 = {A.V[0], A.V[1], A.V[2]};
    uniform float<3> S1 = {B.V[0], B.V[1], B.V[2]};
    uniform float<3> D = S0 * S1;

    uniform FVector Result;
    Result.V[0] = D[0];
    Result.V[1] = D[1];
    Result.V[2] = D[2];
    //  Result.V[3] = D[3];
    return Result;
}

export void TestUniform1(uniform FVector Dst[], const uniform FVector Src0[], const uniform FVector Src1[],
                         const uniform int Count) {
    for (uniform int i = 0; i < Count; i++) {
        Dst[i] = Mul1(Src0[i], Src1[i]);
    }
}

export void TestUniform2(uniform FVector Dst[], const uniform FVector Src0[], const uniform FVector Src1[],
                         const uniform int Count) {
    for (uniform int i = 0; i < Count; i++) {
        Dst[i] = Mul2(Src0[i], Src1[i]);
    }
}

export void TestUniform3(uniform FVector Dst[], const uniform FVector Src0[], const uniform FVector Src1[],
                         const uniform int Count) {
    for (uniform int i = 0; i < Count; i++) {
        Dst[i] = Mul3(Src0[i], Src1[i]);
    }
}
