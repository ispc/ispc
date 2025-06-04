#include "test_static.isph"
struct S {
    unsigned int32 value;
};

// Assignment operator
inline S &operator=(S &a, S b) {
    a.value = b.value + 1; // Adding 1 to the value for demonstration
    return a;
}

// Multiplication assignment operator
inline S &operator*=(S &a, S b) {
    a.value *= b.value;
    return a;
}

// Division assignment operator
inline S &operator/=(S &a, S b) {
    a.value /= b.value;
    return a;
}

// Modulo assignment operator
inline S &operator%=(S &a, S b) {
    a.value %= b.value;
    return a;
}

// Addition assignment operator
inline S &operator+=(S &a, S b) {
    a.value += b.value;
    return a;
}

// Subtraction assignment operator
inline S &operator-=(S &a, S b) {
    a.value -= b.value;
    return a;
}

// Left shift assignment operator
inline S &operator<<=(S &a, S b) {
    a.value <<= b.value;
    return a;
}

// Right shift assignment operator
inline S &operator>>=(S &a, S b) {
    a.value >>= b.value;
    return a;
}

// Bitwise AND assignment operator
inline S &operator&=(S &a, S b) {
    a.value &= b.value;
    return a;
}

// Bitwise XOR assignment operator
inline S &operator^=(S &a, S b) {
    a.value ^= b.value;
    return a;
}

// Bitwise OR assignment operator
inline S &operator|=(S &a, S b) {
    a.value |= b.value;
    return a;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;
    S x, y, z;
    x.value = 100;
    y.value = 5;

    // Assignment
    // Using standard assignment
    S t = x; // t.value = 100
    // Using assignment operator
    z = x; // z.value = 101

    // Multiplication assignment
    z *= y; // z.value = 505 (101 * 5)

    // Division assignment
    S div = z;
    div /= y; // div.value = 101 (505 / 5)

    // Modulo assignment
    S mod;
    mod.value = 17;
    mod %= y; // mod.value = 2 (17 % 5)

    // Addition assignment
    S sum = x;
    sum += y; // sum.value = 105 (100 + 5)

    // Subtraction assignment
    S diff = x;
    diff -= y; // diff.value = 95 (100 - 5)

    // Left shift assignment
    S lshift = x;
    S shift_amount;
    shift_amount.value = 3;
    lshift <<= shift_amount; // lshift.value = 800 (100 << 3)

    // Right shift assignment
    S rshift;
    rshift.value = 64;
    rshift >>= shift_amount; // rshift.value = 8 (64 >> 3)

    // Bitwise AND assignment
    S band;
    band.value = 0xFF;
    S mask;
    mask.value = 0x0F;
    band &= mask; // band.value = 15 (0xFF & 0x0F = 0x0F = 15)

    // Bitwise XOR assignment
    S bxor;
    bxor.value = 0xAA; // 10101010 in binary
    S xmask;
    xmask.value = 0x55; // 01010101 in binary
    bxor ^= xmask;      // bxor.value = 255 (0xAA ^ 0x55 = 0xFF = 255)

    // Bitwise OR assignment
    S bor;
    bor.value = 0xF0; // 11110000 in binary
    S ormask;
    ormask.value = 0x0F; // 00001111 in binary
    bor |= ormask;       // bor.value = 255 (0xF0 | 0x0F = 0xFF = 255)

    // Chaining operations
    S chain;
    chain.value = 10;
    chain += y; // chain.value = 15
    chain *= y; // chain.value = 75
    chain /= y; // chain.value = 15

    RET[0] = extract(z.value, 0);     // Multiplication result
    RET[1] = extract(sum.value, 0);   // Addition result
    RET[2] = extract(diff.value, 0);  // Subtraction result
    RET[3] = extract(bor.value, 0);   // Bitwise OR result
    RET[4] = extract(chain.value, 0); // Bitwise OR result

    // NEW: Test with references as lvalues
    print("Starting reference tests\n");

    // Creating reference variables to test with
    S ref_base, ref_op;
    ref_base.value = 50;
    ref_op.value = 10;

    // References to structures
    S &ref_x = ref_base;
    S &ref_y = ref_op;

    // Assignment with reference as lvalue
    ref_x = x; // ref_x.value = 101 (100 + 1)

    // Multiplication assignment with reference
    ref_x *= y; // ref_x.value = 505 (101 * 5)

    // Division assignment with reference
    S &ref_div = ref_x; // ref pointing to ref_x (value = 505)
    ref_div /= ref_y;   // ref_div.value = 50 (505 / 10)

    // Modulo assignment with reference
    S mod_ref;
    mod_ref.value = 27;
    S &ref_mod = mod_ref;
    ref_mod %= ref_y; // ref_mod.value = 7 (27 % 10)

    // Addition assignment with reference
    S &ref_sum = ref_x; // ref pointing to ref_x (value = 50)
    ref_sum += ref_y;   // ref_sum.value = 60 (50 + 10)

    // Subtraction assignment with reference
    S &ref_diff = ref_x; // ref pointing to ref_x (value = 60)
    ref_diff -= ref_y;   // ref_diff.value = 50 (60 - 10)

    // Left shift assignment with reference
    S lshift_ref;
    lshift_ref.value = 8;
    S shift_ref;
    shift_ref.value = 2;
    S &ref_lshift = lshift_ref;
    S &ref_shift = shift_ref;
    ref_lshift <<= ref_shift; // ref_lshift.value = 32 (8 << 2)

    // Right shift assignment with reference
    S rshift_ref;
    rshift_ref.value = 32;
    S &ref_rshift = rshift_ref;
    ref_rshift >>= ref_shift; // ref_rshift.value = 8 (32 >> 2)

    // Bitwise AND assignment with reference
    S band_ref, mask_ref;
    band_ref.value = 0xCC; // 11001100 in binary
    mask_ref.value = 0xAA; // 10101010 in binary
    S &ref_band = band_ref;
    S &ref_mask = mask_ref;
    ref_band &= ref_mask; // ref_band.value = 136 (0xCC & 0xAA = 0x88 = 136)

    // Bitwise XOR assignment with reference
    S bxor_ref;
    bxor_ref.value = 0x33; // 00110011 in binary
    S &ref_bxor = bxor_ref;
    ref_bxor ^= ref_mask; // ref_bxor.value = 153 (0x33 ^ 0xAA = 0x99 = 153)

    // Bitwise OR assignment with reference
    S bor_ref;
    bor_ref.value = 0x33; // 00110011 in binary
    S &ref_bor = bor_ref;
    ref_bor |= ref_mask; // ref_bor.value = 187 (0x33 | 0xAA = 0xBB = 187)

    // Chain of operations with references
    S chain_ref;
    chain_ref.value = 20;
    S &ref_chain = chain_ref;
    ref_chain += ref_y; // ref_chain.value = 30 (20 + 10)
    ref_chain *= ref_y; // ref_chain.value = 300 (30 * 10)
    ref_chain /= ref_y; // ref_chain.value = 30 (300 / 10)

    // Store results from reference operations
    RET[0] += extract(ref_x.value, 0);      // 50
    RET[1] += extract(ref_mod.value, 0);    // 7
    RET[2] += extract(ref_lshift.value, 0); // 32
    RET[3] += extract(ref_band.value, 0);   // 136
    RET[4] += extract(ref_chain.value, 0);  // 30
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 555;
    RET[1] = 112;
    RET[2] = 127;
    RET[3] = 391;
    RET[4] = 45;
}
