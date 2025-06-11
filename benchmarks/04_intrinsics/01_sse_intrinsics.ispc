// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
#include "intrinsics/xmmintrin.isph"

export void mm_add_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_add_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_add_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_add_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_sub_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_sub_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_sub_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_sub_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_mul_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_mul_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_mul_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_mul_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_div_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_div_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_div_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_div_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_sqrt_ss(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 R = _mm_sqrt_ss(S1);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_sqrt_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 R = _mm_sqrt_ps(S1);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_rcp_ss(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 R = _mm_rcp_ss(S1);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_rcp_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 R = _mm_rcp_ps(S1);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_rsqrt_ss(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 R = _mm_rsqrt_ss(S1);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_rsqrt_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 R = _mm_rsqrt_ps(S1);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_min_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_min_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_min_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_min_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_max_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_max_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_max_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_max_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_and_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_and_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_andnot_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_andnot_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_or_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                     const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_or_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_xor_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_xor_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_cmpeq_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_cmpeq_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_cmpeq_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_cmpeq_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_cmplt_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_cmplt_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_cmplt_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_cmplt_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_cmple_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_cmple_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_cmple_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_cmple_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_cmpgt_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_cmpgt_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_cmpgt_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                        const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_cmpgt_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_cmpnle_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_cmpnle_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_cmpnle_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_cmpnle_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_cmpngt_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_cmpngt_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_cmpngt_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_cmpngt_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_cmpnge_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_cmpnge_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_cmpnge_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_cmpnge_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_cmpord_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_cmpord_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_cmpord_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_cmpord_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_cmpunord_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        const uniform __m128 R = _mm_cmpunord_ss(S1, S2);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_cmpunord_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_cmpunord_ps(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_comieq_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_comieq_ss(S1, S2);
    }
}

export void mm_comilt_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_comilt_ss(S1, S2);
    }
}

export void mm_comile_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_comile_ss(S1, S2);
    }
}

export void mm_comigt_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_comigt_ss(S1, S2);
    }
}

export void mm_comige_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_comige_ss(S1, S2);
    }
}

export void mm_comineq_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_comineq_ss(S1, S2);
    }
}

export void mm_ucomieq_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_ucomieq_ss(S1, S2);
    }
}

export void mm_ucomilt_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_ucomilt_ss(S1, S2);
    }
}

export void mm_ucomile_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_ucomile_ss(S1, S2);
    }
}

export void mm_ucomigt_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_ucomigt_ss(S1, S2);
    }
}

export void mm_ucomige_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_ucomige_ss(S1, S2);
    }
}

export void mm_ucomineq_ss(uniform int Result[], const uniform float Source1[], const uniform float Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 S2 = _mm_load_ss(&Source2[k]);
        Result[k] = _mm_ucomineq_ss(S1, S2);
    }
}

export void mm_cvt_ss2si(uniform int Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform int R = _mm_cvt_ss2si(S1);
        Result[k] = R;
    }
}

export void mm_cvtt_ss2si(uniform int Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform int R = _mm_cvtt_ss2si(S1);
        Result[k] = R;
    }
}

export void mm_cvt_si2ss(uniform float Result[], const uniform float Source1[], const uniform int Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform __m128 R = _mm_cvt_si2ss(S1, Source2[k]);
        _mm_store_ss(&Result[k], R);
    }
}

export void mm_cvtss_f32(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        const uniform float R = _mm_cvtss_f32(S1);
        Result[k] = R;
    }
}

export void mm_shuffle_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                          const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 S = _mm_add_ps(S1, S2);
        const uniform __m128 R = _mm_shuffle_ps(S, S, 0x1B);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_unpackhi_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 S = _mm_add_ps(S1, S2);
        const uniform __m128 R = _mm_unpackhi_ps(S, S);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_unpacklo_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                           const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 S = _mm_add_ps(S1, S2);
        const uniform __m128 R = _mm_unpacklo_ps(S, S);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_movehl_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 S = _mm_add_ps(S1, S2);
        const uniform __m128 R = _mm_movehl_ps(S, S);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_movelh_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                         const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 S = _mm_add_ps(S1, S2);
        const uniform __m128 R = _mm_movelh_ps(S, S);
        _mm_storeu_ps(&Result[k], R);
    }
}

export void mm_movemask_ps(uniform int Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        Result[k] = _mm_movemask_ps(S1);
    }
}

export void mm_set_ss(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_set_ss(Source1[k]);
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_set_ps1(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_set_ps1(Source1[k]);
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_set_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                      const uniform float Source3[], const uniform float Source4[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_set_ps(Source1[k], Source2[k], Source3[k], Source4[k]);
        _mm_storeu_ps(&Result[4 * k], S1);
    }
}

export void mm_setr_ps(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                       const uniform float Source3[], const uniform float Source4[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_setr_ps(Source1[k], Source2[k], Source3[k], Source4[k]);
        _mm_storeu_ps(&Result[4 * k], S1);
    }
}

export void mm_setzero_ps(uniform float Result[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_setzero_ps();
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_load_ss(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        _mm_store_ss(&Result[k], S1);
    }
}

export void mm_load_ps1(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_load_ps1(&Source1[k]);
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_load_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_load_ps(&Source1[k]);
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_loadr_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadr_ps(&Source1[k]);
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_loadu_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_store_ss(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k++) {
        const uniform __m128 S1 = _mm_load_ss(&Source1[k]);
        _mm_store_ss(&Result[k], S1);
    }
}

export void mm_store_ps1(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        _mm_store_ps1(&Result[k], S1);
    }
}

export void mm_store_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        _mm_store_ps(&Result[k], S1);
    }
}

export void mm_storer_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        _mm_storer_ps(&Result[k], S1);
    }
}

export void mm_storeu_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_prefetch_nt(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        _mm_prefetch((const uniform int8 *uniform) & Source1[k + 64], _MM_HINT_NTA);
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_prefetch_t0(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        _mm_prefetch((const uniform int8 *uniform) & Source1[k + 64], _MM_HINT_T0);
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_prefetch_t1(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        _mm_prefetch((const uniform int8 *uniform) & Source1[k + 64], _MM_HINT_T1);
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_prefetch_t2(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        _mm_prefetch((const uniform int8 *uniform) & Source1[k + 64], _MM_HINT_T2);
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        _mm_storeu_ps(&Result[k], S1);
    }
}

export void mm_stream_ps(uniform float Result[], const uniform float Source1[], const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        _mm_stream_ps(&Result[k], S1);
    }
    memory_barrier();
}

export void mm_move_ss(uniform float Result[], const uniform float Source1[], const uniform float Source2[],
                       const uniform int Iterations) {
    for (uniform int k = 0; k < Iterations; k += 4) {
        const uniform __m128 S1 = _mm_loadu_ps(&Source1[k]);
        const uniform __m128 S2 = _mm_loadu_ps(&Source2[k]);
        const uniform __m128 R = _mm_move_ss(S1, S2);
        _mm_storeu_ps(&Result[k], R);
    }
}
