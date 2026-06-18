// RUN: %{ispc} %s --target=sse2-i32x4 -I %S/../../stdlib/include -O2 --nowrap --emit-llvm-text -o - | FileCheck %s
// REQUIRES: X86_ENABLED

#include "intrinsics/emmintrin.isph"

constexpr uniform int shuffle_ps_mask() { return _MM_SHUFFLE(3, 2, 1, 0); }
constexpr uniform int shuffle_pd_mask() { return _MM_SHUFFLE2(1, 0); }
constexpr uniform int prefetch_hint() { return _MM_HINT_T1; }
constexpr uniform int lane() { return 4; }
constexpr uniform int shift() { return 3; }

export uniform double shuffle_pd_constexpr_template() {
    uniform __m128d a = _mm_setr_pd(1.0, 2.0);
    uniform __m128d b = _mm_setr_pd(3.0, 4.0);
    uniform __m128d r = _mm_shuffle_pd<shuffle_pd_mask()>(a, b);
    return r.m128d_f64[1];
}

// CHECK-LABEL: define double @shuffle_pd_constexpr_template()
// CHECK: ret double 4.000000e+00

export void prefetch_constexpr_template(uniform int8 const *uniform p) { _mm_prefetch<prefetch_hint()>(p); }

// CHECK-LABEL: define void @prefetch_constexpr_template(ptr
// CHECK: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 2, i32 1)

export void sse2_constexpr_template_smoke(uniform float *uniform fp, uniform double *uniform dp,
                                          uniform __m128i *uniform ip, uniform int8 const *uniform cp) {
    uniform __m128 a = _mm_loadu_ps(fp);
    uniform __m128 b = _mm_loadu_ps(fp + 4);
    uniform __m128 c = _mm_shuffle_ps<shuffle_ps_mask()>(a, b);
    _mm_storeu_ps(fp, c);

    uniform __m128d da = _mm_loadu_pd(dp);
    uniform __m128d db = _mm_loadu_pd(dp + 2);
    uniform __m128d dc = _mm_shuffle_pd<shuffle_pd_mask()>(da, db);
    _mm_storeu_pd(dp, dc);

    uniform __m128i x = _mm_loadu_si128(ip);
    uniform int e = _mm_extract_epi16<lane()>(x);
    x = _mm_insert_epi16<lane()>(x, e);
    x = _mm_slli_si128<shift()>(x);
    x = _mm_srli_si128<shift()>(x);
    x = _mm_slli_epi16<shift()>(x);
    x = _mm_slli_epi32<shift()>(x);
    x = _mm_slli_epi64<shift()>(x);
    x = _mm_srai_epi16<shift()>(x);
    x = _mm_srai_epi32<shift()>(x);
    x = _mm_srli_epi16<shift()>(x);
    x = _mm_srli_epi32<shift()>(x);
    x = _mm_srli_epi64<shift()>(x);
    x = _mm_shuffle_epi32<shuffle_ps_mask()>(x);
    x = _mm_shufflehi_epi16<shuffle_ps_mask()>(x);
    x = _mm_shufflelo_epi16<shuffle_ps_mask()>(x);
    _mm_storeu_si128(ip, x);

    _mm_prefetch<prefetch_hint()>(cp);
}
