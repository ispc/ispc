export uniform int width() { return programCount; }

// Change layout from
// input: a0 b0 c0 d0 a1 b1 c1 d1 a2 b2 c2 d2 ...
// to
// output a0 a1 ... aX b0 b1 ... bX c0 c1 ... cX d0 d1 .. dX aX+1 ...
// where X = programCount - 1
// input parameter "n" is number of elements input array. Must be multiple of 4*programCount;

#define AOS_TO_SOA4_STDLIB(T)                                                                                          \
    export void aos_to_soa4_stdlib_##T(uniform T *uniform input, uniform T *uniform output, uniform int n) {           \
        uniform int chunk = 4 * programCount;                                                                          \
        uniform int iterations = n / chunk;                                                                            \
                                                                                                                       \
        varying T *uniform output_varying = (varying T * uniform) output;                                              \
                                                                                                                       \
        for (uniform int i = 0; i < iterations; i++) {                                                                 \
            aos_to_soa4(&input[chunk * i], &output_varying[i * 4], &output_varying[i * 4 + 1],                         \
                        &output_varying[i * 4 + 2], &output_varying[i * 4 + 3]);                                       \
        }                                                                                                              \
    }

AOS_TO_SOA4_STDLIB(int32)
AOS_TO_SOA4_STDLIB(float)
AOS_TO_SOA4_STDLIB(int64)
AOS_TO_SOA4_STDLIB(double)

// Change layout from
// input: a0 b0 c0 a1 b1 c1 a2 b2 c2 ...
// to
// output a0 a1 ... aX b0 b1 ... bX c0 c1 ... cX aX+1 ...
// where X = programCount - 1
// input parameter "n" is number of elements input array. Must be multiple of 3*programCount;

#define AOS_TO_SOA3_STDLIB(T)                                                                                          \
    export void aos_to_soa3_stdlib_##T(uniform T *uniform input, uniform T *uniform output, uniform int n) {           \
        uniform int chunk = 3 * programCount;                                                                          \
        uniform int iterations = n / chunk;                                                                            \
                                                                                                                       \
        varying T *uniform output_varying = (varying T * uniform) output;                                              \
                                                                                                                       \
        for (uniform int i = 0; i < iterations; i++) {                                                                 \
            aos_to_soa3(&input[chunk * i], &output_varying[i * 3], &output_varying[i * 3 + 1],                         \
                        &output_varying[i * 3 + 2]);                                                                   \
        }                                                                                                              \
    }

AOS_TO_SOA3_STDLIB(int32)
AOS_TO_SOA3_STDLIB(float)
AOS_TO_SOA3_STDLIB(int64)
AOS_TO_SOA3_STDLIB(double)

// Change layout from
// input: a0 b0 a1 b1 a2 b2 ...
// to
// output a0 a1 ... aX b0 b1 ... bX aX+1 ...
// where X = programCount - 1
// input parameter "n" is number of elements input array. Must be multiple of 2*programCount;

#define AOS_TO_SOA2_STDLIB(T)                                                                                          \
    export void aos_to_soa2_stdlib_##T(uniform T *uniform input, uniform T *uniform output, uniform int n) {           \
        uniform int chunk = 2 * programCount;                                                                          \
        uniform int iterations = n / chunk;                                                                            \
                                                                                                                       \
        varying T *uniform output_varying = (varying T * uniform) output;                                              \
                                                                                                                       \
        for (uniform int i = 0; i < iterations; i++) {                                                                 \
            aos_to_soa2(&input[chunk * i], &output_varying[i * 2], &output_varying[i * 2 + 1]);                        \
        }                                                                                                              \
    }

AOS_TO_SOA2_STDLIB(int32)
AOS_TO_SOA2_STDLIB(float)
AOS_TO_SOA2_STDLIB(int64)
AOS_TO_SOA2_STDLIB(double)

#if TARGET_WIDTH == 4
static const varying int vPerm40 = {0, 4, 1, 5};
static const varying int vPerm41 = {2, 6, 3, 7};
static const varying int vPerm42 = {0, 1, 4, 5};
static const varying int vPerm43 = {2, 3, 6, 7};
#elif TARGET_WIDTH == 8
static const varying int vPerm40 = {0, 4, 8, 12, 1, 5, 9, 13};
static const varying int vPerm41 = {2, 6, 10, 14, 3, 7, 11, 15};
static const varying int vPerm42 = {0, 1, 2, 3, 8, 9, 10, 11};
static const varying int vPerm43 = {4, 5, 6, 7, 12, 13, 14, 15};
#elif TARGET_WIDTH == 16
static const varying int vPerm40 = {0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29};
static const varying int vPerm41 = {2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31};
static const varying int vPerm42 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
static const varying int vPerm43 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
#elif TARGET_WIDTH == 32
static const varying int vPerm40 = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                                    1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61};
static const varying int vPerm41 = {2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,
                                    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63};
static const varying int vPerm42 = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
static const varying int vPerm43 = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
#elif TARGET_WIDTH == 64
static const varying int vPerm40 = {0,  4,  8,  12, 16, 20, 24, 28, 32, 36,  40,  44,  48,  52,  56,  60,
                                    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
                                    1,  5,  9,  13, 17, 21, 25, 29, 33, 37,  41,  45,  49,  53,  57,  61,
                                    65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125};
static const varying int vPerm41 = {2,  6,  10, 14, 18, 22, 26, 30, 34, 38,  42,  46,  50,  54,  58,  62,
                                    66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126,
                                    3,  7,  11, 15, 19, 23, 27, 31, 35, 39,  43,  47,  51,  55,  59,  63,
                                    67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127};
static const varying int vPerm42 = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                                    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
static const varying int vPerm43 = {32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
                                    48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
                                    96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                                    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127};
#else
#error "No implementation for this target"
#endif

#define AOS_TO_SOA4_ISPC(T)                                                                                            \
    static inline void aos_to_soa4_ispc(uniform T src[], varying T *uniform v0, varying T *uniform v1,                 \
                                        varying T *uniform v2, varying T *uniform v3) {                                \
                                                                                                                       \
        const varying T src0 = src[programIndex];                                                                      \
        const varying T src1 = src[programCount + programIndex];                                                       \
        const varying T src2 = src[2 * programCount + programIndex];                                                   \
        const varying T src3 = src[3 * programCount + programIndex];                                                   \
                                                                                                                       \
        const varying T t0 = shuffle(src2, src3, vPerm40);                                                             \
        const varying T t1 = shuffle(src2, src3, vPerm41);                                                             \
        const varying T t2 = shuffle(src0, src1, vPerm40);                                                             \
        const varying T t3 = shuffle(src0, src1, vPerm41);                                                             \
                                                                                                                       \
        *v0 = shuffle(t2, t0, vPerm42);                                                                                \
        *v1 = shuffle(t2, t0, vPerm43);                                                                                \
        *v2 = shuffle(t3, t1, vPerm42);                                                                                \
        *v3 = shuffle(t3, t1, vPerm43);                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    export void aos_to_soa4_ispc_##T(uniform T *uniform input, uniform T *uniform output, uniform int n) {             \
        uniform int chunk = 4 * programCount;                                                                          \
        uniform int iterations = n / chunk;                                                                            \
                                                                                                                       \
        varying T *uniform output_varying = (varying T * uniform) output;                                              \
                                                                                                                       \
        for (uniform int i = 0; i < iterations; i++) {                                                                 \
            aos_to_soa4_ispc(&input[chunk * i], &output_varying[i * 4], &output_varying[i * 4 + 1],                    \
                             &output_varying[i * 4 + 2], &output_varying[i * 4 + 3]);                                  \
        }                                                                                                              \
    }

AOS_TO_SOA4_ISPC(int32)
AOS_TO_SOA4_ISPC(float)
AOS_TO_SOA4_ISPC(int64)
AOS_TO_SOA4_ISPC(double)

#if TARGET_WIDTH == 4
static const varying int vPerm30 = {0, 3, 1, 4};
static const varying int vPerm31 = {2, 5, 3, 6};
static const varying int vPerm32 = {0, 1, 4, 5};
static const varying int vPerm33 = {2, 3, 6, 7};
static const varying int vPerm34 = {2, 5, -1, -1};
static const varying int vPerm35 = {0, 1, 4, 7};
#elif TARGET_WIDTH == 8
static const varying int vPerm30 = {0, 3, 6, 9, 1, 4, 7, 10};
static const varying int vPerm31 = {4, 7, 10, 13, 5, 8, 11, 14};
static const varying int vPerm32 = {0, 1, 2, 3, 8, 9, 10, 11};
static const varying int vPerm33 = {4, 5, 6, 7, 12, 13, 14, 15};
static const varying int vPerm34 = {2, 5, 8, 11, 14, -1, -1, -1};
static const varying int vPerm35 = {0, 1, 2, 3, 4, 9, 12, 15};
#elif TARGET_WIDTH == 16
static const varying int vPerm30 = {0, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22};
static const varying int vPerm31 = {8, 11, 14, 17, 20, 23, 26, 29, 9, 12, 15, 18, 21, 24, 27, 30};
static const varying int vPerm32 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
static const varying int vPerm33 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
static const varying int vPerm34 = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, -1, -1, -1, -1, -1, -1};
static const varying int vPerm35 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31};
#elif TARGET_WIDTH == 32
static const varying int vPerm30 = {0, 3, 6, 9,  12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
                                    1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46};
static const varying int vPerm31 = {16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61,
                                    17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62};
static const varying int vPerm32 = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
static const varying int vPerm33 = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
static const varying int vPerm34 = {2,  5,  8,  11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47,
                                    50, 53, 56, 59, 62, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
static const varying int vPerm35 = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                    16, 17, 18, 19, 20, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63};
#elif TARGET_WIDTH == 64
static const varying int vPerm30 = {0,  3,  6,  9,  12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
                                    48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93,
                                    1,  4,  7,  10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
                                    49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94};
static const varying int vPerm31 = {32, 35, 38, 41, 44, 47, 50, 53,  56,  59,  62,  65,  68,  71,  74,  77,
                                    80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125,
                                    33, 36, 39, 42, 45, 48, 51, 54,  57,  60,  63,  66,  69,  72,  75,  78,
                                    81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126};
static const varying int vPerm32 = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                                    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
static const varying int vPerm33 = {32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
                                    48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
                                    96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                                    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127};
static const varying int vPerm34 = {2,  5,   8,   11,  14,  17,  20,  23,  26,  29,  32, 35, 38, 41, 44, 47,
                                    50, 53,  56,  59,  62,  65,  68,  71,  74,  77,  80, 83, 86, 89, 92, 95,
                                    98, 101, 104, 107, 110, 113, 116, 119, 122, 125, -1, -1, -1, -1, -1, -1,
                                    -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, -1, -1, -1, -1, -1};
static const varying int vPerm35 = {0,  1,  2,  3,  4,  5,  6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
                                    16, 17, 18, 19, 20, 21, 22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
                                    32, 33, 34, 35, 36, 37, 38,  39,  40,  41,  64,  67,  70,  73,  76,  79,
                                    82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127};

#else
#error "No implementation for this target"
#endif

#define AOS_TO_SOA3_ISPC(T)                                                                                            \
    static inline void aos_to_soa3_ispc(uniform T src[], varying T *uniform v0, varying T *uniform v1,                 \
                                        varying T *uniform v2) {                                                       \
                                                                                                                       \
        const varying T src0 = src[programIndex];                                                                      \
        const varying T src1 = src[programCount + programIndex];                                                       \
        const varying T src2 = src[2 * programCount + programIndex];                                                   \
                                                                                                                       \
        const varying T t0 = shuffle(src0, src1, vPerm30);                                                             \
        const varying T t1 = shuffle(src1, src2, vPerm31);                                                             \
        *v0 = shuffle(t0, t1, vPerm32);                                                                                \
        *v1 = shuffle(t0, t1, vPerm33);                                                                                \
                                                                                                                       \
        const varying T t2 = shuffle(src0, src1, vPerm34);                                                             \
        *v2 = shuffle(t2, src2, vPerm35);                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    export void aos_to_soa3_ispc_##T(uniform T *uniform input, uniform T *uniform output, uniform int n) {             \
        uniform int chunk = 3 * programCount;                                                                          \
        uniform int iterations = n / chunk;                                                                            \
                                                                                                                       \
        varying T *uniform output_varying = (varying T * uniform) output;                                              \
                                                                                                                       \
        for (uniform int i = 0; i < iterations; i++) {                                                                 \
            aos_to_soa3_ispc(&input[chunk * i], &output_varying[i * 3], &output_varying[i * 3 + 1],                    \
                             &output_varying[i * 3 + 2]);                                                              \
        }                                                                                                              \
    }

AOS_TO_SOA3_ISPC(int32)
AOS_TO_SOA3_ISPC(float)
AOS_TO_SOA3_ISPC(int64)
AOS_TO_SOA3_ISPC(double)

///////////////////////////////////////////////
#if TARGET_WIDTH == 4
static const varying int vPerm20 = {0, 2, 4, 6};
static const varying int vPerm21 = {1, 3, 5, 7};
#elif TARGET_WIDTH == 8
static const varying int vPerm20 = {0, 2, 4, 6, 8, 10, 12, 14};
static const varying int vPerm21 = {1, 3, 5, 7, 9, 11, 13, 15};
#elif TARGET_WIDTH == 16
static const varying int vPerm20 = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
static const varying int vPerm21 = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
#elif TARGET_WIDTH == 32
static const varying int vPerm20 = {0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                                    32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62};
static const varying int vPerm21 = {1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                                    33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63};
#elif TARGET_WIDTH == 64
static const varying int vPerm20 = {0,  2,  4,   6,   8,   10,  12,  14,  16,  18,  20,  22,  24,  26,  28,  30,
                                    32, 34, 36,  38,  40,  42,  44,  46,  48,  50,  52,  54,  56,  58,  60,  62,
                                    64, 66, 68,  70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,
                                    96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126};
static const varying int vPerm21 = {1,  3,  5,   7,   9,   11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  31,
                                    33, 35, 37,  39,  41,  43,  45,  47,  49,  51,  53,  55,  57,  59,  61,  63,
                                    65, 67, 69,  71,  73,  75,  77,  79,  81,  83,  85,  87,  89,  91,  93,  95,
                                    97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127};
#else
#error "No implementation for this target"
#endif

#define AOS_TO_SOA2_ISPC(T)                                                                                            \
    static inline void aos_to_soa2_ispc(uniform T src[], varying T *uniform v0, varying T *uniform v1) {               \
                                                                                                                       \
        const varying T src0 = src[programIndex];                                                                      \
        const varying T src1 = src[programCount + programIndex];                                                       \
                                                                                                                       \
        *v0 = shuffle(src0, src1, vPerm20);                                                                            \
        *v1 = shuffle(src0, src1, vPerm21);                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    export void aos_to_soa2_ispc_##T(uniform T *uniform input, uniform T *uniform output, uniform int n) {             \
        uniform int chunk = 2 * programCount;                                                                          \
        uniform int iterations = n / chunk;                                                                            \
                                                                                                                       \
        varying T *uniform output_varying = (varying T * uniform) output;                                              \
                                                                                                                       \
        for (uniform int i = 0; i < iterations; i++) {                                                                 \
            aos_to_soa2_ispc(&input[chunk * i], &output_varying[i * 2], &output_varying[i * 2 + 1]);                   \
        }                                                                                                              \
    }

AOS_TO_SOA2_ISPC(int32)
AOS_TO_SOA2_ISPC(float)
AOS_TO_SOA2_ISPC(int64)
AOS_TO_SOA2_ISPC(double)
