// [int32, int64] x [uniform, varying] x [all, even]
#define op(arg) popcnt(arg)
#define popcnt8(arg) op(op(op(op(op(op(op(op(arg))))))))

export void popcnt_uniform_int32_all(uniform int32 *uniform src, uniform int32 *uniform dst, uniform count) {
    for (uniform int i = 0; i < count; i++) {
        dst[i] = popcnt8(src[i]);
    }
}

export void popcnt_uniform_int32_even(uniform int32 *uniform src, uniform int32 *uniform dst, uniform count) {
    for (uniform int i = 0; i < count; i += 2) {
        dst[i] = popcnt8(src[i]);
    }
}

export void popcnt_varying_int32_all(uniform int32 *uniform src, uniform int32 *uniform dst, uniform count) {
    foreach (i = 0...count) { dst[i] = popcnt8(src[i]); }
}

export void popcnt_varying_int32_even(uniform int32 *uniform src, uniform int32 *uniform dst, uniform count) {
    foreach (i = 0...count) {
        if ((programIndex & 1) == 0)
            dst[i] = popcnt8(src[i]);
    }
}

export void popcnt_uniform_int64_all(uniform int64 *uniform src, uniform int64 *uniform dst, uniform count) {
    for (uniform int i = 0; i < count; i++) {
        dst[i] = popcnt8(src[i]);
    }
}

export void popcnt_uniform_int64_even(uniform int64 *uniform src, uniform int64 *uniform dst, uniform count) {
    for (uniform int i = 0; i < count; i += 2) {
        dst[i] = popcnt8(src[i]);
    }
}

export void popcnt_varying_int64_all(uniform int64 *uniform src, uniform int64 *uniform dst, uniform count) {
    foreach (i = 0...count) { dst[i] = popcnt8(src[i]); }
}

export void popcnt_varying_int64_even(uniform int64 *uniform src, uniform int64 *uniform dst, uniform count) {
    foreach (i = 0...count) {
        if ((programIndex & 1) == 0)
            dst[i] = popcnt8(src[i]);
    }
}
