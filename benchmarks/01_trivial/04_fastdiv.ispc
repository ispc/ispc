
#define FASTDIVISPC(UT_ISPC, ST_ISPC, DIV_VAL)                                                                         \
    export void fastdiv_##ST_ISPC##_##DIV_VAL(uniform ST_ISPC *uniform dst, uniform int count) {                       \
        foreach (i = 0... 2 * count) {                                                                                 \
            ST_ISPC num = i - count;                                                                                   \
            dst[i] = num / DIV_VAL;                                                                                    \
        }                                                                                                              \
    }                                                                                                                  \
    export void fastdiv_##UT_ISPC##_##DIV_VAL(uniform UT_ISPC *uniform dst, uniform int count) {                       \
        foreach (i = 0 ... count) {                                                                                    \
            UT_ISPC num = i;                                                                                           \
            dst[i] = num / DIV_VAL;                                                                                    \
        }                                                                                                              \
    }

FASTDIVISPC(uint64, int64, 13)

FASTDIVISPC(uint32, int32, 13)

FASTDIVISPC(uint16, int16, 13)

FASTDIVISPC(uint8, int8, 13)

FASTDIVISPC(uint64, int64, 16)

FASTDIVISPC(uint32, int32, 16)

FASTDIVISPC(uint16, int16, 16)

FASTDIVISPC(uint8, int8, 16)
