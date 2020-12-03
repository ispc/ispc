export uniform int width() { return programCount; }

// This case loads based on condition.
#define PACKEDLOAD(T_ISPC)                                                                                             \
    export uniform uint32 packed_load_active_##T_ISPC##_eq(uniform T_ISPC *uniform src, uniform T_ISPC *uniform dst,   \
                                                           uniform const uint index, uniform int count) {              \
        uniform int32 num = 0;                                                                                         \
        uint32 val = 0;                                                                                                \
        foreach (i = 0...count) {                                                                                      \
            varying T_ISPC var = 0;                                                                                    \
            if ((programIndex & index) == 0)                                                                           \
                num += packed_load_active(&src[num], &var);                                                            \
            val += var;                                                                                                \
        }                                                                                                              \
        return reduce_add(val);                                                                                        \
    }                                                                                                                  \
                                                                                                                       \
    export uniform uint32 packed_load_active_##T_ISPC##_neq(uniform T_ISPC *uniform src, uniform T_ISPC *uniform dst,  \
                                                            uniform const uint index, uniform int count) {             \
        uniform int32 num = 0;                                                                                         \
        uint32 val = 0;                                                                                                \
        foreach (i = 0...count) {                                                                                      \
            varying T_ISPC var = 0;                                                                                    \
            if ((programIndex & index) != 0)                                                                           \
                num += packed_load_active(&src[num], &var);                                                            \
            val += var;                                                                                                \
        }                                                                                                              \
        return reduce_add(val);                                                                                        \
    }

// This case stores based on condition.
#define PACKEDSTORE(T_ISPC, STORE_TYPE)                                                                                \
    export uniform uint32 STORE_TYPE##_##T_ISPC##_eq(uniform T_ISPC *uniform src, uniform T_ISPC *uniform dst,         \
                                                     uniform const uint index, uniform int count) {                    \
        uniform int32 num = 0;                                                                                         \
        foreach (i = 0...count) {                                                                                      \
            varying T_ISPC var = src[i];                                                                               \
            if ((programIndex & index) == 0)                                                                           \
                num += STORE_TYPE(&dst[num], var);                                                                     \
        }                                                                                                              \
        return num;                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    export uniform uint32 STORE_TYPE##_##T_ISPC##_neq(uniform T_ISPC *uniform src, uniform T_ISPC *uniform dst,        \
                                                      uniform const uint index, uniform int count) {                   \
        uniform int32 num = 0;                                                                                         \
        foreach (i = 0...count) {                                                                                      \
            varying T_ISPC var = src[i];                                                                               \
            if ((programIndex & index) != 0)                                                                           \
                num += STORE_TYPE(&dst[num], var);                                                                     \
        }                                                                                                              \
        return num;                                                                                                    \
    }

PACKEDLOAD(int32)
PACKEDSTORE(int32, packed_store_active)
PACKEDSTORE(int32, packed_store_active2)

PACKEDLOAD(int64)
PACKEDSTORE(int64, packed_store_active)
PACKEDSTORE(int64, packed_store_active2)
