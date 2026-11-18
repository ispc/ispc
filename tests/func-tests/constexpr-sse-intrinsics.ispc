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
    uniform __m128 ordinary_f = _mm_shuffle_ps(a, b, reverse_lanes);
    uniform __m128d ordinary_d = _mm_shuffle_pd(da, db, _MM_SHUFFLE2(0, 1));
    return f.m128_f32[0] == offset + 4 && f.m128_f32[1] == offset + 3 && f.m128_f32[2] == offset + 6 &&
           f.m128_f32[3] == offset + 5 && d.m128d_f64[0] == offset + 2 && d.m128d_f64[1] == offset + 3 &&
           ordinary_f.m128_f32[0] == f.m128_f32[0] && ordinary_f.m128_f32[1] == f.m128_f32[1] &&
           ordinary_f.m128_f32[2] == f.m128_f32[2] && ordinary_f.m128_f32[3] == f.m128_f32[3] &&
           ordinary_d.m128d_f64[0] == d.m128d_f64[0] && ordinary_d.m128d_f64[1] == d.m128d_f64[1];
}

uniform bool integer_shuffles(uniform int offset) {
    uniform __m128i words =
        _mm_setr_epi16(offset + 1, offset + 2, offset + 3, offset + 4, offset + 5, offset + 6, offset + 7, offset + 8);
    uniform __m128i reversed_high =
        _mm_setr_epi16(offset + 1, offset + 2, offset + 3, offset + 4, offset + 8, offset + 7, offset + 6, offset + 5);
    uniform __m128i reversed_low =
        _mm_setr_epi16(offset + 4, offset + 3, offset + 2, offset + 1, offset + 5, offset + 6, offset + 7, offset + 8);
    uniform __m128i ints = _mm_setr_epi32(offset + 1, offset + 2, offset + 3, offset + 4);
    uniform __m128i reversed_ints = _mm_setr_epi32(offset + 4, offset + 3, offset + 2, offset + 1);
    return equal_m128i(_mm_shufflehi_epi16<reverse_lanes>(words), reversed_high) &&
           equal_m128i(_mm_shufflelo_epi16<reverse_lanes>(words), reversed_low) &&
           equal_m128i(_mm_shuffle_epi32<reverse_lanes>(ints), reversed_ints) &&
           equal_m128i(_mm_shufflehi_epi16(words, reverse_lanes), reversed_high) &&
           equal_m128i(_mm_shufflelo_epi16(words, reverse_lanes), reversed_low) &&
           equal_m128i(_mm_shuffle_epi32(ints, reverse_lanes), reversed_ints);
}

uniform bool dynamic_shuffles(uniform unsigned int mask) {
    uniform __m128 a = _mm_setr_ps(1, 2, 3, 4);
    uniform __m128 b = _mm_setr_ps(5, 6, 7, 8);
    uniform __m128 f = _mm_shuffle_ps(a, b, mask);
    uniform __m128d da = _mm_setr_pd(1, 2);
    uniform __m128d db = _mm_setr_pd(3, 4);
    uniform __m128d d = _mm_shuffle_pd(da, db, mask);
    uniform __m128i ints = _mm_setr_epi32(-1, -2, -3, -4);
    uniform __m128i i = _mm_shuffle_epi32(ints, mask);
    uniform __m128i words = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
    uniform __m128i hi = _mm_shufflehi_epi16(words, mask);
    uniform __m128i lo = _mm_shufflelo_epi16(words, mask);
    for (uniform int lane = 0; lane < 4; ++lane) {
        uniform int index = (mask >> (lane * 2)) & 3;
        if (i.m128i_i32[lane] != -1 - index || f.m128_f32[lane] != (lane < 2 ? 1 : 5) + index ||
            _mm_extract_epi16(hi, lane) != 1 + lane || _mm_extract_epi16(hi, lane + 4) != 5 + index ||
            _mm_extract_epi16(lo, lane) != 1 + index || _mm_extract_epi16(lo, lane + 4) != 5 + lane) {
            return false;
        }
    }
    return d.m128d_f64[0] == 1 + (mask & 1) && d.m128d_f64[1] == 3 + ((mask >> 1) & 1);
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

// Keep each case in a separate call to bound the number of temporaries at -O0.
template <int Count> static noinline uniform bool equivalent_shifts(uniform __m128i v, uniform int count) {
    return equal_m128i(_mm_slli_si128(v, count), _mm_slli_si128<Count>(v)) &&
           equal_m128i(_mm_srli_si128(v, count), _mm_srli_si128<Count>(v)) &&
           equal_m128i(_mm_slli_epi16(v, count), _mm_slli_epi16<Count>(v)) &&
           equal_m128i(_mm_slli_epi32(v, count), _mm_slli_epi32<Count>(v)) &&
           equal_m128i(_mm_slli_epi64(v, count), _mm_slli_epi64<Count>(v)) &&
           equal_m128i(_mm_srli_epi16(v, count), _mm_srli_epi16<Count>(v)) &&
           equal_m128i(_mm_srli_epi32(v, count), _mm_srli_epi32<Count>(v)) &&
           equal_m128i(_mm_srli_epi64(v, count), _mm_srli_epi64<Count>(v)) &&
           equal_m128i(_mm_srai_epi16(v, count), _mm_srai_epi16<Count>(v)) &&
           equal_m128i(_mm_srai_epi32(v, count), _mm_srai_epi32<Count>(v));
}

uniform bool shift_boundaries(uniform int seed) {
    // Include nonzero bytes at both ends and mixed signed lanes.
    uniform __m128i v = {{0x09080700 | seed, 0x7ffe8001, 0x22334455, 0x80ff00ff}};
    return equivalent_shifts<0>(v, 0) && equivalent_shifts<1>(v, 1) && equivalent_shifts<7>(v, 7) &&
           equivalent_shifts<8>(v, 8) && equivalent_shifts<9>(v, 9) && equivalent_shifts<15>(v, 15) &&
           equivalent_shifts<16>(v, 16) && equivalent_shifts<31>(v, 31) && equivalent_shifts<32>(v, 32) &&
           equivalent_shifts<63>(v, 63) && equivalent_shifts<64>(v, 64) && equivalent_shifts<255>(v, 255) &&
           equivalent_shifts<256>(v, 256) && equivalent_shifts<-1>(v, -1);
}

uniform bool vector_count_shifts(uniform int seed) {
    uniform __m128i words = _mm_setr_epi32(0, -seed, 1, seed);
    uniform __m128i count = _mm_set_epi64x(-1, seed);
    uniform __m128i too_large = _mm_set_epi64x(0, (uniform int64)seed << 32);
    uniform __m128i nonzero_dwords = _mm_setr_epi32(0, -1, -1, -1);
    return equal_m128i(_mm_sll_epi16(words, count), _mm_slli_epi16(words, seed)) &&
           equal_m128i(_mm_sll_epi32(words, count), _mm_slli_epi32(words, seed)) &&
           equal_m128i(_mm_sll_epi64(words, count), _mm_slli_epi64(words, seed)) &&
           equal_m128i(_mm_sra_epi16(words, count), _mm_srai_epi16(words, seed)) &&
           equal_m128i(_mm_sra_epi32(words, count), _mm_srai_epi32(words, seed)) &&
           equal_m128i(_mm_srl_epi16(words, count), _mm_srli_epi16(words, seed)) &&
           equal_m128i(_mm_srl_epi32(words, count), _mm_srli_epi32(words, seed)) &&
           equal_m128i(_mm_srl_epi64(words, count), _mm_srli_epi64(words, seed)) &&
           equal_m128i(_mm_sll_epi16(words, too_large), _mm_setzero_si128()) &&
           equal_m128i(_mm_sra_epi32(words, too_large), nonzero_dwords) &&
           equal_m128i(_mm_srl_epi32(words, too_large), nonzero_dwords);
}

uniform bool insert_extract(uniform int offset) {
    uniform __m128i data = _mm_setr_epi16(offset, 1, 2, 3, 4, 5, 6, -1);
    uniform __m128i inserted = _mm_insert_epi16<7>(data, 0x1beef);
    return _mm_extract_epi16<7>(data) == 65535 && _mm_extract_epi16<7>(inserted) == 0xbeef &&
           _mm_extract_epi16<8>(inserted) == offset && _mm_extract_epi16(data, 7) == 65535 &&
           _mm_extract_epi16(inserted, 7) == 0xbeef && _mm_extract_epi16(inserted, 8) == offset &&
           equal_m128i(_mm_insert_epi16(data, 0x1beef, 7), inserted) &&
           equal_m128i(_mm_insert_epi16<8>(data, 0x1ffff), _mm_setr_epi16(-1, 1, 2, 3, 4, 5, 6, -1)) &&
           equal_m128i(_mm_insert_epi16(data, 0x1ffff, 8), _mm_setr_epi16(-1, 1, 2, 3, 4, 5, 6, -1));
}

uniform bool packed_lanes(uniform int seed) {
    uniform __m128i a = _mm_setr_epi16(INT16_MIN, -(uniform int16)199, -seed, 0, seed, 126, 128, 300);
    uniform __m128i b = _mm_setr_epi16(300, -300, 0, seed, 255, 256, -1, INT16_MAX);
    uniform __m128i signed_expected =
        _mm_setr_epi8(INT8_MIN, INT8_MIN, -seed, 0, seed, 126, INT8_MAX, INT8_MAX, INT8_MAX, INT8_MIN, 0, seed,
                     INT8_MAX, INT8_MAX, -1, INT8_MAX);
    uniform __m128i unsigned_expected =
        _mm_setr_epi8(0, 0, 0, 0, seed, 126, INT8_MIN, -1, -1, 0, 0, seed, -1, -1, 0, -1);
    uniform __m128i dwords_a = _mm_setr_epi32(INT32_MIN, -seed, seed, INT32_MAX);
    uniform __m128i dwords_b = _mm_setr_epi32(65536, seed + 3, -seed - 4, -65536);
    uniform __m128i words_expected =
        _mm_setr_epi16(INT16_MIN, -seed, seed, INT16_MAX, INT16_MAX, seed + 3, -seed - 4, INT16_MIN);
    uniform __m128 packed_float = _mm_castsi128_ps(_mm_setr_epi32(INT32_MIN, seed, 0x7f800135, -seed));
    uniform __m128d packed_double = _mm_castsi128_pd(_mm_set_epi64x(INT64_MIN | seed, seed));
    return equal_m128i(_mm_packs_epi16(a, b), signed_expected) &&
           equal_m128i(_mm_packus_epi16(a, b), unsigned_expected) &&
           equal_m128i(_mm_packs_epi32(dwords_a, dwords_b), words_expected) &&
           _mm_movemask_epi8(signed_expected) == 0x4207 &&
           _mm_movemask_ps(packed_float) == 9 && _mm_movemask_pd(packed_double) == 2;
}

static noinline uniform bool compare_loaded_dwords(uniform __m128i input[]) {
    uniform __m128i lhs = input[0];
    uniform __m128i rhs = input[1];
    return equal_m128i(_mm_cmpeq_epi32(lhs, rhs), _mm_setr_epi32(-1, 0, 0, 0)) &&
           equal_m128i(_mm_cmpgt_epi32(lhs, rhs), _mm_setr_epi32(0, -1, 0, -1)) &&
           equal_m128i(_mm_cmplt_epi32(lhs, rhs), _mm_setr_epi32(0, 0, -1, 0));
}

task void f_f(uniform float RET[], uniform float input[]) {
    uniform int seed = 1 + ((uniform int)input[0] & 3);
    uniform __m128i compare_inputs[2] = {_mm_setr_epi32(INT32_MIN, seed, -seed, INT32_MAX),
                                        _mm_setr_epi32(INT32_MIN, -seed, seed, seed)};
    uniform bool ok = float_shuffles(seed) && integer_shuffles(seed) && byte_shifts(seed) && bit_shifts(seed) &&
                      shift_boundaries(seed) && vector_count_shifts(seed) && insert_extract(seed) && packed_lanes(seed) &&
                      compare_loaded_dwords(compare_inputs) &&
                      dynamic_shuffles(0x100u | (0x49u * seed));
    RET[programIndex] = ok ? 1.f : 0.f;
}

task void result(uniform float RET[]) { RET[programIndex] = 1.f; }
