// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// The lane representation remains uniform even when its short-vector width equals the gang width.
// RUN: %{ispc} %s --target=sse2-i32x4 -I %S/../../stdlib/include --werror --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 -I %S/../../stdlib/include --werror --nowrap --emit-llvm-text -o %t.ll
// RUN: %{ispc} %s --target=avx512icl-x16 -I %S/../../stdlib/include --werror --nowrap --emit-asm -o - | FileCheck %s --check-prefix=PACK
// RUN: %{ispc} %s --target=avx512icl-x32 -I %S/../../stdlib/include --werror --nowrap --emit-llvm-text -o %t.ll
// RUN: %{ispc} %s --target=avx512icl-x64 -I %S/../../stdlib/include --werror --nowrap --emit-asm -o - | FileCheck %s --check-prefix=CMPMASK
// RUN: %{ispc} %s --target=avx512icl-x64 -I %S/../../stdlib/include --werror --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=MASKIR
// REQUIRES: X86_ENABLED

#include "intrinsics/emmintrin.isph"

constexpr uniform __m128i bytes = _mm_setr_epi8(-1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
constexpr uniform __m128i bytes_reverse = _mm_set_epi8(-1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
constexpr uniform __m128i byte_splat = _mm_set1_epi8(-2);
static_assert(bytes.m128i_i32[0] == 0x040302ff && bytes.m128i_i32[3] == 0x100f0e0d);
static_assert(bytes_reverse.m128i_i32[0] == 0x0d0e0f10 && (uniform uint32)bytes_reverse.m128i_i32[3] == 0xff020304u);
static_assert((uniform uint32)byte_splat.m128i_i32[2] == 0xfefefefeu);

constexpr uniform __m128i words = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, -1);
constexpr uniform __m128i words_reverse = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, -1);
constexpr uniform __m128i word_splat = _mm_set1_epi16(-2);
static_assert(_mm_extract_epi16(words, 7) == 65535 && _mm_extract_epi16<8>(words_reverse) == 65535);
static_assert(_mm_extract_epi16(word_splat, 3) == 65534 && _mm_extract_epi16(words_reverse, -1) == 1);

constexpr uniform int reverse = _MM_SHUFFLE(0, 1, 2, 3);
constexpr uniform __m128i high = _mm_shufflehi_epi16(words, reverse);
constexpr uniform __m128i low = _mm_shufflelo_epi16<reverse>(words);
constexpr uniform __m128i inserted = _mm_insert_epi16(high, 0x1beef, 9);
static_assert(_mm_extract_epi16(high, 0) == 1 && _mm_extract_epi16(high, 4) == 65535 && _mm_extract_epi16(high, 7) == 5);
static_assert(_mm_extract_epi16(low, 0) == 4 && _mm_extract_epi16(low, 3) == 1 && _mm_extract_epi16(low, 7) == 65535);
static_assert(_mm_extract_epi16(inserted, 1) == 0xbeef && _mm_extract_epi16(_mm_insert_epi16<9>(high, 7), 1) == 7);

constexpr uniform int64 payload = 0x123456789abcdef0ll;
constexpr uniform __m128i qwords = _mm_set_epi64x(-1, payload);
constexpr uniform __m128i qword_splat = _mm_set1_epi64x(payload);
static_assert(_mm_cvtsi128_si64(qwords) == payload && qwords.m128i_i32[2] == -1 && qwords.m128i_i32[3] == -1);
static_assert(_mm_cvtsi128_si64(_mm_cvtsi64_si128(payload)) == payload && qword_splat.m128i_i32[3] == 0x12345678);
static_assert(_mm_cvtsi128_si32(_mm_cvtsi32_si128(-7)) == -7);
static_assert(_mm_move_epi64(qwords).m128i_i32[2] == 0 && _mm_cvtsi128_si64(_mm_move_epi64(qwords)) == payload);
static_assert(_mm_unpackhi_epi64(qwords, qword_splat).m128i_i32[3] == 0x12345678);
static_assert(_mm_unpacklo_epi32(qwords, qword_splat).m128i_i32[3] == 0x12345678);

constexpr uniform __m128i left_bytes = _mm_slli_si128(bytes, 1);
constexpr uniform __m128i right_bytes = _mm_srli_si128<1>(bytes);
static_assert(left_bytes.m128i_i32[0] == 0x0302ff00 && right_bytes.m128i_i32[0] == 0x05040302);
static_assert(_mm_slli_si128<1>(bytes).m128i_i32[2] == left_bytes.m128i_i32[2]);
static_assert(_mm_slli_si128(bytes, -1).m128i_i32[0] == 0 && _mm_srli_si128(bytes, 16).m128i_i32[3] == 0);
static_assert(_mm_slli_si128(bytes, 7).m128i_i32[2] == 0x05040302);
static_assert(_mm_slli_si128(bytes, 8).m128i_i32[2] == 0x040302ff);
static_assert(_mm_slli_si128(bytes, 9).m128i_i32[2] == 0x0302ff00);
static_assert(_mm_srli_si128(bytes, 7).m128i_i32[0] == 0x0b0a0908);
static_assert(_mm_srli_si128(bytes, 8).m128i_i32[0] == 0x0c0b0a09);
static_assert(_mm_srli_si128(bytes, 9).m128i_i32[0] == 0x0d0c0b0a);

constexpr uniform __m128i three = _mm_set_epi64x(-1, 3);
constexpr uniform __m128i oversized = _mm_set_epi64x(0, 0x100000000ll);
static_assert(_mm_extract_epi16(_mm_slli_epi16(words, 3), 1) == 16);
static_assert(_mm_extract_epi16(_mm_srai_epi16<16>(words), 7) == 65535 &&
              _mm_extract_epi16(_mm_srai_epi16(words, -1), 1) == 0);
static_assert(_mm_slli_epi32(_mm_set1_epi32(1), 31).m128i_i32[0] == INT32_MIN);
static_assert(_mm_srai_epi32(qwords, 32).m128i_i32[3] == -1);
static_assert(_mm_srli_epi32(qwords, 32).m128i_i32[3] == 0 && _mm_srli_epi64(_mm_set1_epi64x(-1), 63).m128i_i32[2] == 1);
static_assert(_mm_cvtsi128_si64(_mm_slli_epi64(_mm_set1_epi64x(1), 31)) == 0x80000000ll);
static_assert(_mm_extract_epi16(_mm_sll_epi16(words, three), 1) == 16);
static_assert(_mm_extract_epi16(_mm_sra_epi16(words, three), 7) == 65535);
static_assert(_mm_extract_epi16(_mm_srl_epi16(words, oversized), 1) == 65535);
static_assert(_mm_cvtsi128_si64(_mm_sll_epi64(qwords, oversized)) == 0);

constexpr uniform __m128i float_bits = _mm_setr_epi32(INT32_MIN, 0x7fc01234, 0, 0x3ff00000);
constexpr uniform __m128 ps = _mm_castsi128_ps(float_bits);
constexpr uniform __m128d pd = _mm_castps_pd(ps);
static_assert(intbits(ps.m128_f32[0]) == 0x80000000u && intbits(ps.m128_f32[1]) == 0x7fc01234u);
static_assert(pd.m128d_f64[1] == 1.0 && _mm_castpd_si128(pd).m128i_i32[1] == 0x7fc01234);
static_assert(_mm_castps_si128(_mm_castpd_ps(pd)).m128i_i32[0] == INT32_MIN);
static_assert(_mm_castsi128_pd(float_bits).m128d_f64[1] == 1.0);
static_assert(intbits(_mm_cvtss_f32(ps)) == 0x80000000u);
static_assert(intbits(_mm_cvtsd_f64(pd)) == 0x7fc0123480000000ull);

// Packed integer arithmetic wraps each lane independently.
static_assert(_mm_add_epi8(_mm_set1_epi8(INT8_MAX), _mm_set1_epi8(1)).m128i_i32[0] == (uniform int32)0x80808080u);
static_assert(_mm_extract_epi16(_mm_add_epi16(_mm_set1_epi16(INT16_MAX), _mm_set1_epi16(1)), 7) == 0x8000);
static_assert(_mm_add_epi32(_mm_set1_epi32(INT32_MAX), _mm_set1_epi32(1)).m128i_i32[0] == INT32_MIN);
static_assert(_mm_cvtsi128_si64(_mm_sub_epi64(_mm_set1_epi64x(INT64_MIN), _mm_set1_epi64x(1))) == INT64_MAX);
static_assert(_mm_extract_epi16(_mm_mullo_epi16(_mm_set1_epi16(-1), _mm_set1_epi16(-1)), 2) == 1);
static_assert(_mm_madd_epi16(_mm_set1_epi16(INT16_MIN), _mm_set1_epi16(INT16_MIN)).m128i_i32[0] == INT32_MIN);

static_assert(_mm_adds_epi8(_mm_set1_epi8(INT8_MAX), _mm_set1_epi8(1)).m128i_i32[3] == 0x7f7f7f7f);
static_assert(_mm_extract_epi16(_mm_subs_epi16(_mm_set1_epi16(INT16_MIN), _mm_set1_epi16(1)), 3) == 0x8000);
static_assert(_mm_adds_epu8(_mm_set1_epi8(-1), _mm_set1_epi8(1)).m128i_i32[2] == -1);
static_assert(_mm_subs_epu16(_mm_setzero_si128(), _mm_set1_epi16(1)).m128i_i32[2] == 0);
static_assert(_mm_avg_epu8(_mm_set1_epi8(-1), _mm_set1_epi8(-2)).m128i_i32[1] == -1);
static_assert(_mm_extract_epi16(_mm_avg_epu16(_mm_set1_epi16(-1), _mm_set1_epi16(-2)), 4) == 65535);

static_assert(_mm_extract_epi16(_mm_max_epi16(words, word_splat), 7) == 65535);
static_assert(_mm_min_epu8(bytes, byte_splat).m128i_i32[0] == 0x040302fe);
static_assert(_mm_cmpeq_epi8(bytes, bytes).m128i_i32[1] == -1);
static_assert(_mm_extract_epi16(_mm_cmpgt_epi16(words, word_splat), 7) == 65535);
static_assert(_mm_cmplt_epi32(_mm_set1_epi32(-1), _mm_setzero_si128()).m128i_i32[2] == -1);
static_assert(_mm_extract_epi16(_mm_mulhi_epi16(_mm_set1_epi16(-1), _mm_set1_epi16(1)), 4) == 65535);
static_assert(_mm_extract_epi16(_mm_mulhi_epu16(_mm_set1_epi16(-1), _mm_set1_epi16(-1)), 4) == 65534);
constexpr uniform __m128i products = _mm_mul_epu32(_mm_setr_epi32(7, 11, 13, 17), _mm_setr_epi32(2, 3, 4, 5));
static_assert(_mm_cvtsi128_si64(products) == 14 && products.m128i_i32[2] == 33 && products.m128i_i32[3] == 0);
constexpr uniform __m128i distances = _mm_sad_epu8(bytes, _mm_setzero_si128());
static_assert(_mm_cvtsi128_si64(distances) == 290 && distances.m128i_i32[2] == 100);

constexpr uniform __m128i pack_a = _mm_setr_epi16(-400, INT8_MIN, -1, 0, 1, 127, 128, 500);
constexpr uniform __m128i pack_b = _mm_setr_epi16(300, -300, 2, 3, 4, 5, 6, 7);
constexpr uniform __m128i signed_bytes = _mm_packs_epi16(pack_a, pack_b);
constexpr uniform __m128i unsigned_bytes = _mm_packus_epi16(pack_a, pack_b);
static_assert(signed_bytes.m128i_i32[0] == 0x00ff8080 && signed_bytes.m128i_i32[2] == 0x0302807f);
static_assert((uniform uint32)unsigned_bytes.m128i_i32[1] == 0xff807f01u && unsigned_bytes.m128i_i32[2] == 0x030200ff);
constexpr uniform __m128i signed_words =
    _mm_packs_epi32(_mm_setr_epi32(-70000, INT16_MIN, INT16_MAX, 70000), _mm_setr_epi32(11, -12, 13, -14));
static_assert((uniform uint32)signed_words.m128i_i32[0] == 0x80008000u && signed_words.m128i_i32[1] == 0x7fff7fff);
static_assert(_mm_extract_epi16(signed_words, 5) == 65524 && _mm_extract_epi16(signed_words, 7) == 65522);
static_assert(_mm_unpacklo_epi8(bytes, byte_splat).m128i_i32[0] == (uniform int32)0xfe02feffu);
static_assert(_mm_extract_epi16(_mm_unpackhi_epi16(words, word_splat), 6) == 65535);

static_assert(_mm_xor_si128(bytes, bytes).m128i_i32[0] == 0);
static_assert(intbits(_mm_and_ps(ps, ps).m128_f32[1]) == 0x7fc01234u);
static_assert(intbits(_mm_or_pd(pd, pd).m128d_f64[0]) == intbits(pd.m128d_f64[0]));
static_assert(intbits(_mm_unpacklo_ps(ps, _mm_setzero_ps()).m128_f32[0]) == 0x80000000u);
static_assert(intbits(_mm_move_ss(_mm_setzero_ps(), ps).m128_f32[0]) == 0x80000000u);
static_assert(intbits(_mm_move_sd(pd, _mm_setzero_pd()).m128d_f64[0]) == 0);
static_assert(_mm_movemask_epi8(bytes) == 1 && _mm_movemask_ps(ps) == 1);
static_assert(_mm_movemask_pd(_mm_castsi128_pd(_mm_set_epi64x(INT64_MIN, 1))) == 2);

// CHECK-LABEL: shift_packed_words___
// CHECK: psllw
void shift_packed_words(uniform __m128i data[], uniform int count) {
    data[0] = _mm_slli_epi16(data[0], count);
}

// CHECK-LABEL: shuffle_high_words___
// CHECK: pshufhw
void shuffle_high_words(uniform __m128i data[]) {
    data[0] = _mm_shufflehi_epi16(data[0], _MM_SHUFFLE(0, 1, 2, 3));
}

// CHECK-LABEL: shift_left_bytes___
// CHECK: pslldq
void shift_left_bytes(uniform __m128i data[]) {
    data[0] = _mm_slli_si128<3>(data[0]);
}

// CHECK-LABEL: mask_packed_bytes___
// CHECK: pmovmskb
uniform int mask_packed_bytes(uniform __m128i data[]) { return _mm_movemask_epi8(data[0]); }

// CHECK-LABEL: add_saturated_words___
// CHECK: paddsw
void add_saturated_words(uniform __m128i data[]) { data[0] = _mm_adds_epi16(data[0], data[1]); }

// PACK-LABEL: pack_signed_dwords___
// PACK: {{vpackssdw|vpmovsdw}}
void pack_signed_dwords(uniform __m128i a[], uniform __m128i b[], uniform __m128i result[]) {
    result[0] = _mm_packs_epi32(a[0], b[0]);
}

// CMPMASK-LABEL: compare_packed_dwords___
// CMPMASK: vpcmpgtd
// CMPMASK-NOT: vpmovm2d
// CMPMASK: ret
// MASKIR-LABEL: @compare_packed_dwords___
// MASKIR: icmp sgt <4 x i32>
void compare_packed_dwords(uniform int32 a[], uniform int32 b[], uniform int32 result[]) {
    const uniform __m128i lhs = {{a[0], a[1], a[2], a[3]}};
    const uniform __m128i rhs = {{b[0], b[1], b[2], b[3]}};
    const uniform __m128i mask = _mm_cmpgt_epi32(lhs, rhs);
    result[0] = mask.m128i_i32[0];
    result[1] = mask.m128i_i32[1];
    result[2] = mask.m128i_i32[2];
    result[3] = mask.m128i_i32[3];
}

// MASKIR-LABEL: @compare_full_dwords___
// MASKIR: icmp sgt <16 x i32>
void compare_full_dwords(uniform int32<16> a[], uniform int32<16> b[], uniform int32<16> result[]) {
    result[0] = -(uniform int32<16>)(a[0] > b[0]);
}
