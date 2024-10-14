// RUN: %{ispc} -O2 --target=avx2-i32x8 --emit-asm --x86-asm-syntax=intel %s -o - 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: vpermpd

struct vec {
    double V[4];
};

struct trans {
    vec X;
    vec Y;
};

inline uniform vec operator+(const uniform vec &A, const uniform vec &B) {
    uniform vec Result;
    foreach(i = 0 ... 4) { Result.V[i] = A.V[i] + B.V[i]; }
    return Result;
}

inline uniform vec VectorSwizzle(const uniform vec &Vec, const uniform int X, const uniform int Y, const uniform int Z, const uniform int W) {
    const uniform vec Result = {{Vec.V[X], Vec.V[Y], Vec.V[Z], Vec.V[W]}};
    return Result;
}


extern "C" uniform vec foo(const uniform trans &T, const uniform vec& V) {
    const uniform vec A = T.X + V;
    const uniform vec B = VectorSwizzle(A, 1,2,0,3);
    const uniform vec C = B + T.Y;
    return C;
}
