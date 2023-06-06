#define MASKED_LOAD_STORE(T_ISPC)                                                                                      \
    export void masked_load_store_##T_ISPC(uniform T_ISPC aIN[], uniform T_ISPC bIN[], uniform T_ISPC cOUT[],          \
                                           uniform int n, uniform bool aMask[]) {                                      \
        bool mask = aMask[programIndex];                                                                               \
        for (int i = 0; i < n; i += programCount) {                                                                    \
            if (mask) {                                                                                                \
                T_ISPC a = aIN[i + programIndex];                                                                      \
                T_ISPC b = bIN[i + programIndex];                                                                      \
                cOUT[i + programIndex] = b - a;                                                                        \
            }                                                                                                          \
        }                                                                                                              \
    }

MASKED_LOAD_STORE(int8)
MASKED_LOAD_STORE(int16)
MASKED_LOAD_STORE(int32)
MASKED_LOAD_STORE(float)
MASKED_LOAD_STORE(double)
MASKED_LOAD_STORE(int64)