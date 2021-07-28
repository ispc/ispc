#include "../test_static.isph"
const int64 bit_size = 64;
const int64 a_bit_size = 3;
const int64 max_a = 7; // must be 2^a_bit_size-1

const int64 max_shift = bit_size - a_bit_size;

static int64 calc_shift(int64 idx)
{
    return (idx * (bit_size / programCount)) % (max_shift + 1);
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int64 a = aFOO[programIndex];
    a = a % (max_a + 1);
    // to make a = 0bXX1
    a = a % 2 == 0 ? 1 : a;
    // If a is just 1, compiler is smart enough to eliminate count_leading_zeros.
    int64 i = a << calc_shift(programIndex);
    RET[programIndex] = count_trailing_zeros(i);
}

task void result(uniform float RET[]) {
    RET[programIndex] = calc_shift(programIndex);
}
