#include "../../ispcrt/ispcrt.isph"

// Macros

#define UNROLL_VARYING_SIG(T_FOR, T_ISPC, UNROLL_FACTOR)                                                               \
    export void unroll_varying_##T_FOR##_##T_ISPC##_##UNROLL_FACTOR(uniform T_ISPC src[], uniform T_ISPC dst[],        \
                                                                    uniform uint64 num_elems)

#define UNROLL_VARYING_FOREACH_BODY                                                                                    \
    foreach (i = 1 ... num_elems) { dst[i] = src[i] + src[i - 1]; }

#define UNROLL_VARYING_FOR_BODY                                                                                        \
    for (uint64 i = programIndex + 1ull; i < num_elems; i += programCount) {                                           \
        dst[i] = src[i] + src[i - 1ull];                                                                               \
    }

// uint64

UNROLL_VARYING_SIG(foreach, uint64, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint64, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, uint64, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint64, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, uint64, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint64, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOR_BODY;
}

// int64

UNROLL_VARYING_SIG(foreach, int64, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int64, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, int64, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int64, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, int64, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int64, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOR_BODY;
}

// uint32

UNROLL_VARYING_SIG(foreach, uint32, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint32, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, uint32, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint32, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, uint32, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint32, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOR_BODY;
}

// int32

UNROLL_VARYING_SIG(foreach, int32, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int32, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, int32, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int32, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, int32, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int32, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOR_BODY;
}

// uint16

UNROLL_VARYING_SIG(foreach, uint16, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint16, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, uint16, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint16, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, uint16, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint16, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOR_BODY;
}

// int16

UNROLL_VARYING_SIG(foreach, int16, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int16, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, int16, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int16, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, int16, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int16, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOR_BODY;
}

// uint8

UNROLL_VARYING_SIG(foreach, uint8, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint8, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, uint8, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint8, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, uint8, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, uint8, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOR_BODY;
}

// int8

UNROLL_VARYING_SIG(foreach, int8, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int8, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, int8, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int8, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, int8, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, int8, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOR_BODY;
}

// double

UNROLL_VARYING_SIG(foreach, double, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, double, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, double, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, double, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, double, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, double, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOR_BODY;
}

// float

UNROLL_VARYING_SIG(foreach, float, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, float, 1) {
    dst[0] = src[0];
#pragma nounroll
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, float, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, float, 2) {
    dst[0] = src[0];
#pragma unroll(2)
    UNROLL_VARYING_FOR_BODY;
}

UNROLL_VARYING_SIG(foreach, float, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOREACH_BODY;
}

UNROLL_VARYING_SIG(for, float, 4) {
    dst[0] = src[0];
#pragma unroll(4)
    UNROLL_VARYING_FOR_BODY;
}
