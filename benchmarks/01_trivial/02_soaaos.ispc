export uniform int width() { return programCount; }

// Change layout from
// input a0 a1 ... aX b0 b1 ... bX c0 c1 ... cX d0 d1 .. dX aX+1 ...
// to
// output: a0 b0 c0 d0 a1 b1 c1 d1 a2 b2 c2 d2 ...
// where X = programCount - 1
// input parameter "n" is number of elements input array. Must be multiple of 4*programCount;

#define SOA_TO_AOS4_STDLIB(T)                                                                                          \
    export void soa_to_aos4_stdlib_##T(uniform T *uniform input, uniform T *uniform output, uniform int n) {           \
        uniform int chunk = 4 * programCount;                                                                          \
        uniform int iterations = n / chunk;                                                                            \
                                                                                                                       \
        varying T *uniform input_varying = (varying T * uniform) input;                                                \
                                                                                                                       \
        for (uniform int i = 0; i < iterations; i++) {                                                                 \
            soa_to_aos4(input_varying[i * 4], input_varying[i * 4 + 1], input_varying[i * 4 + 2],                      \
                        input_varying[i * 4 + 3], &output[chunk * i]);                                                 \
        }                                                                                                              \
    }

SOA_TO_AOS4_STDLIB(int32)
SOA_TO_AOS4_STDLIB(float)
SOA_TO_AOS4_STDLIB(int64)
SOA_TO_AOS4_STDLIB(double)

// Change layout from
// input a0 a1 ... aX b0 b1 ... bX c0 c1 ... cX aX+1 ...
// to
// output: a0 b0 c0 a1 b1 c1 a2 b2 c2 ...
// where X = programCount - 1
// input parameter "n" is number of elements input array. Must be multiple of 3*programCount;

#define SOA_TO_AOS3_STDLIB(T)                                                                                          \
    export void soa_to_aos3_stdlib_##T(uniform T *uniform input, uniform T *uniform output, uniform int n) {           \
        uniform int chunk = 3 * programCount;                                                                          \
        uniform int iterations = n / chunk;                                                                            \
                                                                                                                       \
        varying T *uniform input_varying = (varying T * uniform) input;                                                \
                                                                                                                       \
        for (uniform int i = 0; i < iterations; i++) {                                                                 \
            soa_to_aos3(input_varying[i * 3], input_varying[i * 3 + 1], input_varying[i * 3 + 2], &output[chunk * i]); \
        }                                                                                                              \
    }

SOA_TO_AOS3_STDLIB(int32)
SOA_TO_AOS3_STDLIB(float)
SOA_TO_AOS3_STDLIB(int64)
SOA_TO_AOS3_STDLIB(double)

// Change layout from
// input a0 a1 ... aX b0 b1 ... bX ...
// to
// output: a0 b0 a1 b1 a2 b2 ...
// where X = programCount - 1
// input parameter "n" is number of elements input array. Must be multiple of 2*programCount;

#define SOA_TO_AOS2_STDLIB(T)                                                                                          \
    export void soa_to_aos2_stdlib_##T(uniform T *uniform input, uniform T *uniform output, uniform int n) {           \
        uniform int chunk = 2 * programCount;                                                                          \
        uniform int iterations = n / chunk;                                                                            \
                                                                                                                       \
        varying T *uniform input_varying = (varying T * uniform) input;                                                \
                                                                                                                       \
        for (uniform int i = 0; i < iterations; i++) {                                                                 \
            soa_to_aos2(input_varying[i * 2], input_varying[i * 2 + 1], &output[chunk * i]);                           \
        }                                                                                                              \
    }

SOA_TO_AOS2_STDLIB(int32)
SOA_TO_AOS2_STDLIB(float)
SOA_TO_AOS2_STDLIB(int64)
SOA_TO_AOS2_STDLIB(double)
