// RUN: %{ispc} %s --target=host --opt=fast-math -O2 --emit-llvm-text -o -| FileCheck %s

// CHECK-COUNT-4: fmul

static inline varying float square(const varying float value)
{
    return value * value;
}

template<int N>
static inline varying float power(const varying float value)
{
    varying float accum = value;

    #pragma unroll
    for (uniform int i = 1; i < N; i++) {
        accum *= value;
    }

    return accum;
}

void power_9(uniform float values[], const uniform int count)
{
    assume(((uniform uint64)((void*)values) & (32 * TARGET_WIDTH)-1) == 0);
    assume(count % TARGET_WIDTH == 0);

    #pragma nounroll
    foreach(index = 0...count) {
        values[index] = power<9>(values[index]);
        // values[index] = square(square(square(values[index]))) * values[index];
    }
}
