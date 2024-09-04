// RUN: %{ispc} --target=host --nowrap --emit-asm -o - %s 2>&1 | FileCheck %s

// CHECK-NOT: Error: Unable to find any matching overload for call

typedef struct {
    uniform float<4> m128_f32;
} __m128;

typedef uniform __m128 VectorRegister4Float;

static inline unmasked uniform __m128 _mm_add_ps(uniform __m128 _A, uniform __m128 _B);

inline uniform VectorRegister4Float VectorAdd(const uniform VectorRegister4Float &Vec1,
                                              const uniform VectorRegister4Float &Vec2) {
    return _mm_add_ps(Vec1, Vec2);
}
