// math:
// sqrt, rsqrt / rsqrt_fast, rcp / rcp_fast, ldexp, frexp,
// sin, asin, cos, acos, sincos, tan, atan, atan2, exp, log, pow

#define SQRT(T)                                                                                                        \
    export void sqrt_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                          \
        foreach (i = 0 ... count) { dst[i] = sqrt(src[i]); }                                                           \
    }

#define RSQRT(T)                                                                                                       \
    export void rsqrt_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                         \
        foreach (i = 0 ... count) { dst[i] = rsqrt(src[i]); }                                                          \
    }

#define RSQRT_FAST(T)                                                                                                  \
    export void rsqrt_fast_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                    \
        foreach (i = 0 ... count) { dst[i] = rsqrt_fast(src[i]); }                                                     \
    }

#define RCP(T)                                                                                                         \
    export void rcp_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                           \
        foreach (i = 0 ... count) { dst[i] = rcp(src[i]); }                                                            \
    }

#define RCP_FAST(T)                                                                                                    \
    export void rcp_fast_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                      \
        foreach (i = 0 ... count) { dst[i] = rcp_fast(src[i]); }                                                       \
    }

#define LDEXP(T)                                                                                                       \
    export void ldexp_##T(uniform T *uniform src1, uniform int *uniform src2, uniform T *uniform dst,                  \
                          uniform int count) {                                                                         \
        foreach (i = 0 ... count) { dst[i] = ldexp(src1[i], src2[i]); }                                                \
    }

#define FREXP(T)                                                                                                       \
    export void frexp_##T(uniform T *uniform src, uniform T *uniform dst1, uniform int *uniform dst2,                  \
                          uniform int count) {                                                                         \
        foreach (i = 0 ... count) {                                                                                    \
            varying int pow = 0;                                                                                       \
            dst1[i] = frexp(src[i], &pow);                                                                             \
            dst2[i] = pow;                                                                                             \
        }                                                                                                              \
    }

#define SIN(T)                                                                                                         \
    export void sin_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                           \
        foreach (i = 0 ... count) { dst[i] = sin(src[i]); }                                                            \
    }

#define ASIN(T)                                                                                                        \
    export void asin_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                          \
        foreach (i = 0 ... count) { dst[i] = asin(src[i]); }                                                           \
    }

#define COS(T)                                                                                                         \
    export void cos_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                           \
        foreach (i = 0 ... count) { dst[i] = cos(src[i]); }                                                            \
    }

#define ACOS(T)                                                                                                        \
    export void acos_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                          \
        foreach (i = 0 ... count) { dst[i] = acos(src[i]); }                                                           \
    }

// Missing sincos() here as it's not universallu available.

#define TAN(T)                                                                                                         \
    export void tan_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                           \
        foreach (i = 0 ... count) { dst[i] = tan(src[i]); }                                                            \
    }

#define ATAN(T)                                                                                                        \
    export void atan_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                          \
        foreach (i = 0 ... count) { dst[i] = atan(src[i]); }                                                           \
    }

#define ATAN2(T)                                                                                                       \
    export void atan2_##T(uniform T *uniform src1, uniform T *uniform src2, uniform T *uniform dst,                    \
                          uniform int count) {                                                                         \
        foreach (i = 0 ... count) { dst[i] = atan2(src1[i], src2[i]); }                                                \
    }

#define EXP(T)                                                                                                         \
    export void exp_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                           \
        foreach (i = 0 ... count) { dst[i] = exp(src[i]); }                                                            \
    }

#define LOG(T)                                                                                                         \
    export void log_##T(uniform T *uniform src, uniform T *uniform dst, uniform int count) {                           \
        foreach (i = 0 ... count) { dst[i] = log(src[i]); }                                                            \
    }

#define POW(T)                                                                                                         \
    export void pow_##T(uniform T *uniform src1, uniform T *uniform src2, uniform T *uniform dst, uniform int count) { \
        foreach (i = 0 ... count) { dst[i] = pow(src1[i], src2[i]); }                                                  \
    }

SQRT(float)
SQRT(double)
RSQRT(float)
RSQRT(double)
RSQRT_FAST(float)
RSQRT_FAST(double)
RCP(float)
RCP(double)
RCP_FAST(float)
RCP_FAST(double)
LDEXP(float)
LDEXP(double)
FREXP(float)
FREXP(double)

SIN(float)
SIN(double)
ASIN(float)
ASIN(double)
COS(float)
COS(double)
ACOS(float)
ACOS(double)
TAN(float)
TAN(double)
ATAN(float)
ATAN(double)
ATAN2(float)
ATAN2(double)

EXP(float)
EXP(double)
LOG(float)
LOG(double)
POW(float)
POW(double)
