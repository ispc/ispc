#include "../test_static.isph"
const int32 bit_size = 32;
const int32 a_bit_size = 3;
const int32 max_a = 7; // must be 2^a_bit_size-1

const int32 max_shift = bit_size - a_bit_size;

static int32 calc_shift(int32 idx)
{
    return (idx * (bit_size / programCount)) % (max_shift + 1);
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 a = aFOO[programIndex];
    a = a % (max_a + 1);
    // to make a = 0bXX1
    a = a % 2 == 0 ? 1 : a;
    // If a is just 1, compiler is smart enough to eliminate count_leading_zeros.
    int32 shift = calc_shift(programIndex);
    int32 i = a << shift;
    if (programIndex % 2 == 0)
    {
        RET[programIndex] = count_trailing_zeros(i);
    }
    else
    {
        RET[programIndex] = 0f;
    }
}

task void result(uniform float RET[]) {
    int32 shift = calc_shift(programIndex);
    if (programIndex % 2 == 0)
    {
        RET[programIndex] = shift;
    }
    else
    {
        RET[programIndex] = 0f;
    }
}
