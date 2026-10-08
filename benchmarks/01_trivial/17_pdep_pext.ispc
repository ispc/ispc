// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// [pdep, pext] x [uint32, uint64] x [uniform, varying] with dynamic masks.
#define BIT_OP(OP, W)                                                                                                  \
    export void OP##W##_uniform(uniform uint##W *uniform src, uniform uint##W *uniform mask,                           \
                                uniform uint##W *uniform dst, uniform int count) {                                     \
        for (uniform int i = 0; i < count; i++) {                                                                      \
            dst[i] = OP##W(src[i], mask[i]);                                                                           \
        }                                                                                                              \
    }                                                                                                                  \
    export void OP##W##_varying(uniform uint##W *uniform src, uniform uint##W *uniform mask,                           \
                                uniform uint##W *uniform dst, uniform int count) {                                     \
        foreach (i = 0 ... count) {                                                                                    \
            dst[i] = OP##W(src[i], mask[i]);                                                                           \
        }                                                                                                              \
    }                                                                                                                  \
    /* Partial execution mask: about half of the lanes are active. */                                                  \
    export void OP##W##_varying_half(uniform uint##W *uniform src, uniform uint##W *uniform mask,                      \
                                     uniform uint##W *uniform dst, uniform int count) {                                \
        foreach (i = 0 ... count) {                                                                                    \
            if ((src[i] & 1) == 0)                                                                                     \
                dst[i] = OP##W(src[i], mask[i]);                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    /* Sparse execution mask: about one lane in eight is active. */                                                    \
    export void OP##W##_varying_sparse(uniform uint##W *uniform src, uniform uint##W *uniform mask,                    \
                                       uniform uint##W *uniform dst, uniform int count) {                              \
        foreach (i = 0 ... count) {                                                                                    \
            if ((src[i] & 7) == 0)                                                                                     \
                dst[i] = OP##W(src[i], mask[i]);                                                                       \
        }                                                                                                              \
    }

BIT_OP(pdep, 32)
BIT_OP(pext, 32)
BIT_OP(pdep, 64)
BIT_OP(pext, 64)

// The same mask as a compile-time constant and as a runtime argument.
#define MASK_OP(OP, W, NAME, MASK)                                                                                     \
    export void OP##W##_##NAME##_const_uniform(uniform uint##W *uniform src, uniform uint##W *uniform dst,             \
                                               uniform int count) {                                                    \
        for (uniform int i = 0; i < count; i++) {                                                                      \
            dst[i] = OP##W(src[i], MASK);                                                                              \
        }                                                                                                              \
    }                                                                                                                  \
    export void OP##W##_##NAME##_const_varying(uniform uint##W *uniform src, uniform uint##W *uniform dst,             \
                                               uniform int count) {                                                    \
        foreach (i = 0 ... count) {                                                                                    \
            dst[i] = OP##W(src[i], MASK);                                                                              \
        }                                                                                                              \
    }                                                                                                                  \
    export void OP##W##_##NAME##_runtime_uniform(uniform uint##W *uniform src, uniform uint##W *uniform dst,           \
                                                 uniform int count, uniform uint##W mask) {                            \
        for (uniform int i = 0; i < count; i++) {                                                                      \
            dst[i] = OP##W(src[i], mask);                                                                              \
        }                                                                                                              \
    }                                                                                                                  \
    export void OP##W##_##NAME##_runtime_varying(uniform uint##W *uniform src, uniform uint##W *uniform dst,           \
                                                 uniform int count, uniform uint##W mask) {                            \
        foreach (i = 0 ... count) {                                                                                    \
            dst[i] = OP##W(src[i], mask);                                                                              \
        }                                                                                                              \
    }

// Morton masks: every other bit.
MASK_OP(pdep, 32, morton, 0x55555555u)
MASK_OP(pext, 32, morton, 0x55555555u)
MASK_OP(pdep, 64, morton, 0x5555555555555555ull)
MASK_OP(pext, 64, morton, 0x5555555555555555ull)

// Byte fields: every other byte.
MASK_OP(pdep, 32, bytes, 0x00FF00FFu)
MASK_OP(pext, 32, bytes, 0x00FF00FFu)
MASK_OP(pdep, 64, bytes, 0x00FF00FF00FF00FFull)
MASK_OP(pext, 64, bytes, 0x00FF00FF00FF00FFull)

// 2D Morton encode/decode of 32-bit coordinates into 64-bit codes, using
// pdep/pext with constant masks, pdep/pext with runtime masks, and shifts.
export void morton_encode_pdep_const(uniform uint32 *uniform x, uniform uint32 *uniform y, uniform uint64 *uniform dst,
                                     uniform int count) {
    foreach (i = 0 ... count) {
        dst[i] = pdep64(x[i], 0x5555555555555555ull) | pdep64(y[i], 0xAAAAAAAAAAAAAAAAull);
    }
}

export void morton_encode_pdep_runtime(uniform uint32 *uniform x, uniform uint32 *uniform y,
                                       uniform uint64 *uniform dst, uniform int count, uniform uint64 mask_x,
                                       uniform uint64 mask_y) {
    foreach (i = 0 ... count) {
        dst[i] = pdep64(x[i], mask_x) | pdep64(y[i], mask_y);
    }
}

static inline uint64 spread_bits(uint64 v) {
    v = (v | (v << 16)) & 0x0000FFFF0000FFFFull;
    v = (v | (v << 8)) & 0x00FF00FF00FF00FFull;
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0Full;
    v = (v | (v << 2)) & 0x3333333333333333ull;
    v = (v | (v << 1)) & 0x5555555555555555ull;
    return v;
}

static inline uint32 compact_bits(uint64 v) {
    v &= 0x5555555555555555ull;
    v = (v | (v >> 1)) & 0x3333333333333333ull;
    v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0Full;
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FFull;
    v = (v | (v >> 8)) & 0x0000FFFF0000FFFFull;
    v = (v | (v >> 16)) & 0x00000000FFFFFFFFull;
    return (uint32)v;
}

export void morton_encode_shifts(uniform uint32 *uniform x, uniform uint32 *uniform y, uniform uint64 *uniform dst,
                                 uniform int count) {
    foreach (i = 0 ... count) {
        dst[i] = spread_bits(x[i]) | (spread_bits(y[i]) << 1);
    }
}

export void morton_decode_pext_const(uniform uint64 *uniform src, uniform uint32 *uniform x, uniform uint32 *uniform y,
                                     uniform int count) {
    foreach (i = 0 ... count) {
        x[i] = (uint32)pext64(src[i], 0x5555555555555555ull);
        y[i] = (uint32)pext64(src[i], 0xAAAAAAAAAAAAAAAAull);
    }
}

export void morton_decode_pext_runtime(uniform uint64 *uniform src, uniform uint32 *uniform x,
                                       uniform uint32 *uniform y, uniform int count, uniform uint64 mask_x,
                                       uniform uint64 mask_y) {
    foreach (i = 0 ... count) {
        x[i] = (uint32)pext64(src[i], mask_x);
        y[i] = (uint32)pext64(src[i], mask_y);
    }
}

export void morton_decode_shifts(uniform uint64 *uniform src, uniform uint32 *uniform x, uniform uint32 *uniform y,
                                 uniform int count) {
    foreach (i = 0 ... count) {
        x[i] = compact_bits(src[i]);
        y[i] = compact_bits(src[i] >> 1);
    }
}
