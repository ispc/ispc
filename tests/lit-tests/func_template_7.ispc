// Check that operator overloading kicks-in for template parameter arguments.
// If it doesn't, the test will crash, because type check will alarm binary operation on non-atomic types.

// RUN: %{ispc} %s --nostdlib --target=host -o %t.o

struct S {
    float<4> V;
};

inline uniform S operator+(const uniform S &A, const uniform S &B) {
    varying float S0, S1, Result;
    *((uniform S * uniform) & S0) = *((uniform S * uniform) & A);
    *((uniform S * uniform) & S1) = *((uniform S * uniform) & B);

    Result = S0 + S1;

    return *((uniform S * uniform) & Result);
}

template <typename T> inline T VectorAdd(const T &A, const T &B) { return A + B; }

export void Add(uniform float A[], uniform float B[], uniform float C[], uniform int i) {
    const uniform S Vb = *((uniform S * uniform) & B[i]);
    const uniform S Vc = *((uniform S * uniform) & C[i]);
    *((uniform S * uniform) & A[i]) = VectorAdd(Vb, Vc);
}
