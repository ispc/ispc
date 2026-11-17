// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "../../stdlib/include/intrinsics/emmintrin.isph"
#include "test_static.isph"

constexpr uniform int reverse_lanes = _MM_SHUFFLE(0, 1, 2, 3);
constexpr uniform int shift_one() { return 1; }

uniform bool equal_m128i(uniform __m128i lhs, uniform __m128i rhs) {
    return lhs.m128i_i32[0] == rhs.m128i_i32[0] && lhs.m128i_i32[1] == rhs.m128i_i32[1] &&
           lhs.m128i_i32[2] == rhs.m128i_i32[2] && lhs.m128i_i32[3] == rhs.m128i_i32[3];
}

uniform bool float_shuffles(uniform int offset) {
    uniform __m128 a = _mm_setr_ps(offset + 1, offset + 2, offset + 3, offset + 4);
    uniform __m128 b = _mm_setr_ps(offset + 5, offset + 6, offset + 7, offset + 8);
    uniform __m128 f = _mm_shuffle_ps<reverse_lanes>(a, b);
    uniform __m128d da = _mm_setr_pd(offset + 1, offset + 2);
    uniform __m128d db = _mm_setr_pd(offset + 3, offset + 4);
    uniform __m128d d = _mm_shuffle_pd<_MM_SHUFFLE2(0, 1)>(da, db);
    return f.m128_f32[0] == offset + 4 && f.m128_f32[1] == offset + 3 && f.m128_f32[2] == offset + 6 &&
           f.m128_f32[3] == offset + 5 && d.m128d_f64[0] == offset + 2 && d.m128d_f64[1] == offset + 3;
}

uniform bool integer_shuffles(uniform int offset) {
    uniform __m128i words =
        _mm_setr_epi16(offset + 1, offset + 2, offset + 3, offset + 4, offset + 5, offset + 6, offset + 7, offset + 8);
    uniform __m128i reversed_high =
        _mm_setr_epi16(offset + 1, offset + 2, offset + 3, offset + 4, offset + 8, offset + 7, offset + 6, offset + 5);
    uniform __m128i reversed_low =
        _mm_setr_epi16(offset + 4, offset + 3, offset + 2, offset + 1, offset + 5, offset + 6, offset + 7, offset + 8);
    uniform __m128i ints = _mm_setr_epi32(offset + 1, offset + 2, offset + 3, offset + 4);
    return equal_m128i(_mm_shufflehi_epi16<reverse_lanes>(words), reversed_high) &&
           equal_m128i(_mm_shufflelo_epi16<reverse_lanes>(words), reversed_low) &&
           equal_m128i(_mm_shuffle_epi32<reverse_lanes>(ints),
                       _mm_setr_epi32(offset + 4, offset + 3, offset + 2, offset + 1));
}

uniform bool byte_shifts(uniform int64 seed) {
    uniform __m128i data = _mm_set_epi64x(seed, seed);
    uniform __m128i zero = _mm_setzero_si128();
    return equal_m128i(_mm_slli_si128<shift_one()>(data), _mm_set_epi64x(seed << 8, seed << 8)) &&
           equal_m128i(_mm_srli_si128<shift_one()>(data), _mm_set_epi64x(0, seed << 56)) &&
           equal_m128i(_mm_slli_si128<15>(data), _mm_set_epi64x(seed << 56, 0)) &&
           equal_m128i(_mm_srli_si128<15>(_mm_set_epi64x(seed << 56, seed)), _mm_set_epi64x(0, seed)) &&
           equal_m128i(_mm_slli_si128<16>(data), zero) && equal_m128i(_mm_srli_si128<16>(data), zero) &&
           equal_m128i(_mm_slli_si128<256>(data), zero) && equal_m128i(_mm_srli_si128<-1>(data), zero);
}

uniform bool bit_shifts(uniform int64 seed) {
    uniform __m128i zero = _mm_setzero_si128();
    uniform __m128i data = _mm_set_epi64x(-2 * seed, 2 * seed);
    uniform __m128i both_words = _mm_set1_epi16((uniform int16)(2 * seed));
    uniform __m128i both_ints = _mm_set1_epi32((uniform int32)(2 * seed));
    uniform __m128i signs_as_words = _mm_setr_epi16(0, 0, 0, 0, -1, -1, -1, -1);
    uniform __m128i signs_as_ints = _mm_setr_epi32(0, 0, -1, -1);
    return equal_m128i(_mm_slli_epi16<shift_one()>(both_words), _mm_set1_epi16((uniform int16)(4 * seed))) &&
           equal_m128i(_mm_slli_epi32<shift_one()>(both_ints), _mm_set1_epi32((uniform int32)(4 * seed))) &&
           equal_m128i(_mm_slli_epi64<shift_one()>(data), _mm_set_epi64x(-4 * seed, 4 * seed)) &&
           equal_m128i(_mm_slli_epi16<256>(both_words), zero) && equal_m128i(_mm_slli_epi32<32>(both_ints), zero) &&
           equal_m128i(_mm_slli_epi64<-1>(data), zero) &&
           equal_m128i(_mm_srli_epi16<shift_one()>(both_words), _mm_set1_epi16((uniform int16)seed)) &&
           equal_m128i(_mm_srli_epi32<shift_one()>(both_ints), _mm_set1_epi32((uniform int32)seed)) &&
           equal_m128i(_mm_srli_epi64<shift_one()>(data), _mm_set_epi64x(0x7fffffffffffffffULL - seed + 1, seed)) &&
           equal_m128i(_mm_srli_epi16<16>(data), zero) && equal_m128i(_mm_srli_epi32<256>(data), zero) &&
           equal_m128i(_mm_srli_epi64<64>(data), zero) && equal_m128i(_mm_srai_epi16<16>(data), signs_as_words) &&
           equal_m128i(_mm_srai_epi32<256>(data), signs_as_ints);
}

uniform bool insert_extract(uniform int offset) {
    uniform __m128i data = _mm_setr_epi16(offset, 1, 2, 3, 4, 5, 6, -1);
    uniform __m128i inserted = _mm_insert_epi16<7>(data, 0x1beef);
    return _mm_extract_epi16<7>(data) == 65535 && _mm_extract_epi16<7>(inserted) == 0xbeef &&
           _mm_extract_epi16<8>(inserted) == offset &&
           equal_m128i(_mm_insert_epi16<8>(data, 0x1ffff), _mm_setr_epi16(-1, 1, 2, 3, 4, 5, 6, -1));
}

task void f_f(uniform float RET[], uniform float input[]) {
    uniform int seed = 1 + ((uniform int)input[0] & 3);
    uniform bool ok =
        float_shuffles(seed) && integer_shuffles(seed) && byte_shifts(seed) && bit_shifts(seed) && insert_extract(seed);
    RET[programIndex] = ok ? 1.f : 0.f;
}

task void result(uniform float RET[]) { RET[programIndex] = 1.f; }
