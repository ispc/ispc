// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
#include "intrinsics/emmintrin.isph"

export void mm_add_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_add_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_add_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_add_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_sub_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_sub_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_sub_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_sub_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_mul_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_mul_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_mul_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_mul_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_sqrt_sd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d R = _mm_sqrt_sd(S1, S1);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_sqrt_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d R = _mm_sqrt_pd(S1);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_div_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_div_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_div_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_div_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_min_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_min_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_min_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_min_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_max_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_max_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_max_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_max_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_and_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_and_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_andnot_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_andnot_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_or_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                     const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_or_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_xor_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_xor_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cmpeq_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_cmpeq_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_cmpeq_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_cmpeq_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cmplt_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_cmplt_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_cmplt_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_cmplt_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cmple_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_cmple_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_cmple_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_cmple_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cmpgt_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_cmpgt_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_cmpgt_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_cmpgt_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cmpnle_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_cmpnle_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_cmpnle_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_cmpnle_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cmpngt_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_cmpngt_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_cmpngt_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_cmpngt_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cmpnge_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_cmpnge_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_cmpnge_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_cmpnge_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cmpord_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_cmpord_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_cmpord_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_cmpord_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cmpunord_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        const uniform __m128d R = _mm_cmpunord_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_cmpunord_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d R = _mm_cmpunord_pd(S1, S2);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_comieq_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_comieq_sd(S1, S2);
    }
}

export void mm_comilt_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_comilt_sd(S1, S2);
    }
}

export void mm_comile_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_comile_sd(S1, S2);
    }
}

export void mm_comigt_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_comigt_sd(S1, S2);
    }
}

export void mm_comige_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_comige_sd(S1, S2);
    }
}

export void mm_comineq_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_comineq_sd(S1, S2);
    }
}

export void mm_ucomieq_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_ucomieq_sd(S1, S2);
    }
}

export void mm_ucomilt_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_ucomilt_sd(S1, S2);
    }
}

export void mm_ucomile_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_ucomile_sd(S1, S2);
    }
}

export void mm_ucomigt_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_ucomigt_sd(S1, S2);
    }
}

export void mm_ucomige_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_ucomige_sd(S1, S2);
    }
}

export void mm_ucomineq_sd(uniform int Result[], const uniform double Source1[], const uniform double Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        const uniform __m128d S2 = _mm_load_sd(&Source2[k]);
        Result[k] = _mm_ucomineq_sd(S1, S2);
    }
}

export void mm_cvtepi32_pd(uniform double Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128d R = _mm_cvtepi32_pd(S1);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cvtpd_epi32(uniform int32 Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128i R = _mm_cvtpd_epi32(S1);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cvttpd_epi32(uniform int32 Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128i R = _mm_cvttpd_epi32(S1);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cvtepi32_ps(uniform float Result[], const uniform int Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_load_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128 R = _mm_cvtepi32_ps(S1);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_cvtps_epi32(uniform int Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128i R = _mm_cvtps_epi32(S1);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cvttps_epi32(uniform int Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128i R = _mm_cvttps_epi32(S1);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cvtpd_ps(uniform float Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128 R = _mm_cvtpd_ps(S1);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_cvtps_pd(uniform double Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128d R = _mm_cvtps_pd(S1);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cvtsd_ss(uniform float Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_setzero_ps();
        const uniform __m128d S2 = _mm_load_sd(&Source1[k]);
        const uniform __m128 R = _mm_cvtsd_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_cvtss_sd(uniform double Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_setzero_pd();
        const uniform __m128 S2 = _mm_load_ss(&Source1[k]);
        const uniform __m128d R = _mm_cvtss_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_cvtsd_si32(uniform int Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        Result[k] = _mm_cvtsd_si32(S1);
    }
}

export void mm_cvttsd_si32(uniform int Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        Result[k] = _mm_cvttsd_si32(S1);
    }
}

export void mm_cvtsi32_sd(uniform double Result[], const uniform int Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_setzero_pd();
        const uniform int S2 = Source1[k];
        const uniform __m128d R = _mm_cvtsi32_sd(S1, S2);
        _mm_store_sd(&Result[k], R);
    }
}

export void mm_unpackhi_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d S = _mm_add_pd(S1, S2);
        const uniform __m128d R = _mm_unpackhi_pd(S, S);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_unpacklo_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d S = _mm_add_pd(S1, S2);
        const uniform __m128d R = _mm_unpacklo_pd(S, S);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_movemask_pd(uniform int Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        Result[k] = _mm_movemask_pd(S1);
    }
}

export void mm_shuffle_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d S = _mm_add_pd(S1, S2);
        const uniform __m128d R = _mm_shuffle_pd(S, S, 0x1);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_load_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_load_pd(&Source1[k]);
        _mm_storeu_pd(&Result[k], S1);
    }
}

export void mm_load1_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_load1_pd(&Source1[k]);
        _mm_storeu_pd(&Result[k], S1);
    }
}

export void mm_loadr_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadr_pd(&Source1[k]);
        _mm_storeu_pd(&Result[k], S1);
    }
}

export void mm_loadu_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        _mm_storeu_pd(&Result[k], S1);
    }
}

export void mm_load_sd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        _mm_store_sd(&Result[k], S1);
    }
}

export void mm_loadh_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_load1_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadh_pd(S1, &Source2[k]);
        const uniform __m128d S = _mm_add_pd(S1, S2);
        _mm_storeu_pd(&Result[k], S);
    }
}

export void mm_loadl_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_load1_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadl_pd(S1, &Source2[k]);
        const uniform __m128d S = _mm_add_pd(S1, S2);
        _mm_storeu_pd(&Result[k], S);
    }
}

export void mm_set_sd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_set_sd(Source1[k]);
        _mm_storeu_pd(&Result[k], S1);
    }
}

export void mm_set1_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_set1_pd(Source1[k]);
        _mm_storeu_pd(&Result[k], S1);
    }
}

export void mm_set_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_set_pd(Source1[k], Source1[k + 1]);
        _mm_storeu_pd(&Result[k], S1);
    }
}

export void mm_setr_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_setr_pd(Source1[k], Source1[k + 1]);
        _mm_storeu_pd(&Result[k], S1);
    }
}

export void mm_setzero_pd(uniform double Result[], const uniform int Iterations) {
    const uniform __m128d S1 = _mm_setzero_pd();
    for (uniform int k = 0; k < Iterations; k += 2) {
        _mm_storeu_pd(&Result[k], S1);
    }
}

export void mm_move_sd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                       const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d S = _mm_add_pd(S1, S2);
        const uniform __m128d R = _mm_move_sd(S1, S);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_store_sd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        _mm_store_sd(&Result[k], S1);
    }
}

export void mm_store1_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_load_sd(&Source1[k]);
        _mm_store1_pd(&Result[k], S1);
    }
}

export void mm_store_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        _mm_store_pd(&Result[k], S1);
    }
}

export void mm_storeu_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        _mm_storeu_pd(&Result[k], S1);
    }
}

export void mm_storer_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        _mm_storer_pd(&Result[k], S1);
    }
}

export void mm_storeh_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d S = _mm_add_pd(S1, S2);
        _mm_storeh_pd(&Result[k], S);
    }
}

export void mm_storel_pd(uniform double Result[], const uniform double Source1[], const uniform double Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d S2 = _mm_loadu_pd(&Source2[k]);
        const uniform __m128d S = _mm_add_pd(S1, S2);
        _mm_storel_pd(&Result[k], S);
    }
}

export void mm_add_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int8 Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_add_epi8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_add_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_add_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_add_epi32(uniform int Result[], const uniform int Source1[], const uniform int Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_add_epi32(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_add_epi64(uniform int64 Result[], const uniform int64 Source1[], const uniform int64 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_add_epi64(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_adds_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int8 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_adds_epi8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_adds_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_adds_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_adds_epu8(uniform uint8 Result[], const uniform uint8 Source1[], const uniform uint8 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_adds_epu8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_adds_epu16(uniform uint16 Result[], const uniform uint16 Source1[], const uniform uint16 Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_adds_epu16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_avg_epu8(uniform uint8 Result[], const uniform uint8 Source1[], const uniform uint8 Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_avg_epu8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_avg_epu16(uniform uint16 Result[], const uniform uint16 Source1[], const uniform uint16 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_avg_epu16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_madd_epi16(uniform int32 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_madd_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_max_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_max_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_max_epu8(uniform uint8 Result[], const uniform uint8 Source1[], const uniform uint8 Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_max_epu8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_min_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_min_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_min_epu8(uniform uint8 Result[], const uniform uint8 Source1[], const uniform uint8 Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_min_epu8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_mulhi_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_mulhi_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_mulhi_epu16(uniform uint16 Result[], const uniform uint16 Source1[], const uniform uint16 Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_mulhi_epu16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_mullo_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_mullo_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_mul_epu32(uniform uint64 Result[], const uniform uint32 Source1[], const uniform uint32 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_mul_epu32(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_sad_epu8(uniform uint16 Result[], const uniform uint8 Source1[], const uniform uint8 Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_sad_epu8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_sub_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int8 Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_sub_epi8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_sub_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_sub_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_sub_epi32(uniform int Result[], const uniform int Source1[], const uniform int Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_sub_epi32(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_sub_epi64(uniform int64 Result[], const uniform int64 Source1[], const uniform int64 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_sub_epi64(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_subs_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int8 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_subs_epi8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_subs_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_subs_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_subs_epu8(uniform uint8 Result[], const uniform uint8 Source1[], const uniform uint8 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_subs_epu8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_subs_epu16(uniform uint16 Result[], const uniform uint16 Source1[], const uniform uint16 Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_subs_epu16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_and_si128(uniform uint32 Result[], const uniform uint32 Source1[], const uniform uint32 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_and_si128(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_andnot_si128(uniform uint32 Result[], const uniform uint32 Source1[], const uniform uint32 Source2[],
                            const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_andnot_si128(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_or_si128(uniform uint32 Result[], const uniform uint32 Source1[], const uniform uint32 Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_or_si128(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_xor_si128(uniform uint32 Result[], const uniform uint32 Source1[], const uniform uint32 Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_xor_si128(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_slli_si128(uniform uint32 Result[], const uniform uint32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_slli_si128(S1, 8);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_slli_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_slli_epi16(S1, 8);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_sll_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int Iterations) {
    uniform __m128i _Count = _mm_setzero_si128();
    _Count.m128i_i32[0] = 8;
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_sll_epi16(S1, _Count);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_slli_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_slli_epi32(S1, 8);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_sll_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    uniform __m128i _Count = _mm_setzero_si128();
    _Count.m128i_i32[0] = 8;
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_sll_epi32(S1, _Count);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_slli_epi64(uniform int64 Result[], const uniform int64 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_slli_epi64(S1, 8);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_sll_epi64(uniform int64 Result[], const uniform int64 Source1[], const uniform int Iterations) {
    uniform __m128i _Count = _mm_setzero_si128();
    _Count.m128i_i32[0] = 8;
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_sll_epi64(S1, _Count);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_srai_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_srai_epi16(S1, 4);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_sra_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int Iterations) {
    uniform __m128i _Count = _mm_setzero_si128();
    _Count.m128i_i32[0] = 4;
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_sra_epi16(S1, _Count);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_srai_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_srai_epi32(S1, 4);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_sra_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    uniform __m128i _Count = _mm_setzero_si128();
    _Count.m128i_i32[0] = 4;
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_sra_epi32(S1, _Count);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_srli_si128(uniform uint32 Result[], const uniform uint32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_srli_si128(S1, 4);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_srli_epi16(uniform uint16 Result[], const uniform uint16 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_srli_epi16(S1, 4);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_srl_epi16(uniform uint16 Result[], const uniform uint16 Source1[], const uniform int Iterations) {
    uniform __m128i _Count = _mm_setzero_si128();
    _Count.m128i_i32[0] = 4;
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_srl_epi16(S1, _Count);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_srli_epi32(uniform uint32 Result[], const uniform uint32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_srli_epi32(S1, 4);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_srl_epi32(uniform uint32 Result[], const uniform uint32 Source1[], const uniform int Iterations) {
    uniform __m128i _Count = _mm_setzero_si128();
    _Count.m128i_i32[0] = 4;
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_srl_epi32(S1, _Count);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_srli_epi64(uniform uint64 Result[], const uniform uint64 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_srli_epi64(S1, 4);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_srl_epi64(uniform uint64 Result[], const uniform uint64 Source1[], const uniform int Iterations) {
    uniform __m128i _Count = _mm_setzero_si128();
    _Count.m128i_i32[0] = 4;
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_srl_epi64(S1, _Count);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cmpeq_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int8 Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_cmpeq_epi8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cmpeq_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_cmpeq_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cmpeq_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int32 Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_cmpeq_epi32(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cmpgt_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int8 Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_cmpgt_epi8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cmpgt_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_cmpgt_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cmpgt_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int32 Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_cmpgt_epi32(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cmplt_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int8 Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_cmplt_epi8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cmplt_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_cmplt_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cmplt_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int32 Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_cmplt_epi32(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_cvtsi32_si128(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_cvtsi32_si128(Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], S1);
    }
}

export void mm_cvtsi128_si32(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        Result[k] = _mm_cvtsi128_si32(S1);
    }
}

export void mm_packs_epi16(uniform int8 Result[], const uniform int16 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k + 8]);
        const uniform __m128i R = _mm_packs_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_packs_epi32(uniform int16 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k + 4]);
        const uniform __m128i R = _mm_packs_epi32(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_packus_epi16(uniform uint8 Result[], const uniform int16 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k + 8]);
        const uniform __m128i R = _mm_packus_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_extract_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        *((uniform int *uniform) & Result[k]) = _mm_extract_epi16(S1, 4);
    }
}

export void mm_insert_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        uniform __m128i R = _mm_insert_epi16(S1, 1, 4);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_movemask_epi8(uniform int32 Result[], const uniform int8 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        uniform int R = _mm_movemask_epi8(S1);
        Result[k] = R;
    }
}

export void mm_shuffle_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int32 Source2[],
                             const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i S = _mm_add_epi32(S1, S2);
        const uniform __m128i R = _mm_shuffle_epi32(S, 0x1B);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_shufflehi_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                               const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i S = _mm_add_epi16(S1, S2);
        const uniform __m128i R = _mm_shufflehi_epi16(S, 0x1B);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_shufflelo_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                               const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i S = _mm_add_epi16(S1, S2);
        const uniform __m128i R = _mm_shufflelo_epi16(S, 0x1B);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_unpackhi_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int8 Source2[],
                             const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_unpackhi_epi8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_unpackhi_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                              const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_unpackhi_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_unpackhi_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int32 Source2[],
                              const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_unpackhi_epi32(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_unpackhi_epi64(uniform int64 Result[], const uniform int64 Source1[], const uniform int64 Source2[],
                              const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_unpackhi_epi64(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_unpacklo_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int8 Source2[],
                             const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_unpacklo_epi8(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_unpacklo_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int16 Source2[],
                              const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_unpacklo_epi16(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_unpacklo_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int32 Source2[],
                              const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_unpacklo_epi32(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_unpacklo_epi64(uniform int64 Result[], const uniform int64 Source1[], const uniform int64 Source2[],
                              const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i S2 = _mm_loadu_si128((uniform __m128i * uniform) & Source2[k]);
        const uniform __m128i R = _mm_unpacklo_epi64(S1, S2);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_load_si128(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_load_si128((uniform __m128i * uniform) & Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], S1);
    }
}

export void mm_loadu_si128(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], S1);
    }
}

export void mm_loadl_epi64(uniform int64 Result[], const uniform int64 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadl_epi64((uniform __m128i * uniform) & Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], S1);
    }
}

export void mm_set_epi64x(uniform int64 Result[], const uniform int64 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i R = _mm_set_epi64x(Source1[k + 1], Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_set_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i R = _mm_set_epi32(Source1[k + 3], Source1[k + 2], Source1[k + 1], Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_set_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i R = _mm_set_epi16(Source1[k + 7], Source1[k + 6], Source1[k + 5], Source1[k + 4],
                                                Source1[k + 3], Source1[k + 2], Source1[k + 1], Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_set_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i R =
            _mm_set_epi8(Source1[k + 15], Source1[k + 14], Source1[k + 13], Source1[k + 12], Source1[k + 11],
                         Source1[k + 10], Source1[k + 9], Source1[k + 8], Source1[k + 7], Source1[k + 6],
                         Source1[k + 5], Source1[k + 4], Source1[k + 3], Source1[k + 2], Source1[k + 1], Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_set1_epi64x(uniform int64 Result[], const uniform int64 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i R = _mm_set1_epi64x(Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_set1_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i R = _mm_set1_epi32(Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_set1_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i R = _mm_set1_epi16(Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_set1_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i R = _mm_set1_epi8(Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_setr_epi32(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i R = _mm_setr_epi32(Source1[k], Source1[k + 1], Source1[k + 2], Source1[k + 3]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_setr_epi16(uniform int16 Result[], const uniform int16 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 8) {
        const uniform __m128i R = _mm_setr_epi16(Source1[k], Source1[k + 1], Source1[k + 2], Source1[k + 3],
                                                 Source1[k + 4], Source1[k + 5], Source1[k + 6], Source1[k + 7]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_setr_epi8(uniform int8 Result[], const uniform int8 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i R =
            _mm_setr_epi8(Source1[k], Source1[k + 1], Source1[k + 2], Source1[k + 3], Source1[k + 4], Source1[k + 5],
                          Source1[k + 6], Source1[k + 7], Source1[k + 8], Source1[k + 9], Source1[k + 10],
                          Source1[k + 11], Source1[k + 12], Source1[k + 13], Source1[k + 14], Source1[k + 15]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_setzero_si128(uniform int32 Result[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i R = _mm_setzero_si128();
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_store_si128(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        _mm_store_si128((uniform __m128i * uniform) & Result[k], S1);
    }
}

export void mm_storeu_si128(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], S1);
    }
}

export void mm_storel_epi64(uniform int64 Result[], const uniform int64 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        _mm_storel_epi64((uniform __m128i * uniform) & Result[k], S1);
    }
}

export void mm_maskmoveu_si128(uniform int8 Result[], const uniform int8 Source1[], const uniform int8 Mask[],
                               const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 16) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i M = _mm_loadu_si128((uniform __m128i * uniform) & Mask[k]);
        _mm_maskmoveu_si128(S1, M, &Result[k]);
    }
}

export void mm_move_epi64(uniform int64 Result[], const uniform int64 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128i R = _mm_move_epi64(S1);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_stream_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_load_pd(&Source1[k]);
        _mm_stream_pd(&Result[k], S1);
    }
    memory_barrier();
}

export void mm_stream_si128(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        _mm_stream_si128((uniform __m128i * uniform) & Result[k], S1);
    }
    memory_barrier();
}

export void mm_stream_si32(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        _mm_stream_si32(&Result[k], Source1[k]);
    }
    memory_barrier();
}

export void mm_cvtsd_f64(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_load_pd(&Source1[k]);
        Result[k] = _mm_cvtsd_f64(S1);
    }
}

export void mm_castpd_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128d S1 = _mm_loadu_pd((uniform double *uniform) & Source1[k]);
        const uniform __m128 R = _mm_castpd_ps(S1);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_castpd_si128(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128d S1 = _mm_loadu_pd((uniform double *uniform) & Source1[k]);
        const uniform __m128i R = _mm_castpd_si128(S1);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_castps_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128 S1 = _mm_loadu_ps((uniform float *uniform) & Source1[k]);
        const uniform __m128d R = _mm_castps_pd(S1);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_castps_si128(uniform int32 Result[], const uniform int32 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps((uniform float *uniform) & Source1[k]);
        const uniform __m128i R = _mm_castps_si128(S1);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], R);
    }
}

export void mm_castsi128_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128 R = _mm_castsi128_ps(S1);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_castsi128_pd(uniform double Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        const uniform __m128d R = _mm_castsi128_pd(S1);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cvtsd_si64(uniform int64 Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        Result[k] = _mm_cvtsd_si64(S1);
    }
}

export void mm_cvttsd_si64(uniform int64 Result[], const uniform double Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        Result[k] = _mm_cvttsd_si64(S1);
    }
}

export void mm_cvtsi64_sd(uniform double Result[], const uniform double Source1[], const uniform int64 Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128d S1 = _mm_loadu_pd(&Source1[k]);
        const uniform __m128d R = _mm_cvtsi64_sd(S1, Source2[k]);
        _mm_storeu_pd(&Result[k], R);
    }
}

export void mm_cvtsi64_si128(uniform int64 Result[], const uniform int64 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_cvtsi64_si128(Source1[k]);
        _mm_storeu_si128((uniform __m128i * uniform) & Result[k], S1);
    }
}

export void mm_cvtsi128_si64(uniform int64 Result[], const uniform int64 Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 2) {
        const uniform __m128i S1 = _mm_loadu_si128((uniform __m128i * uniform) & Source1[k]);
        Result[k] = _mm_cvtsi128_si64(S1);
    }
}
