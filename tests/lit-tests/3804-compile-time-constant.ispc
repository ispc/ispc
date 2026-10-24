// #3804: an `inline` function is force-inlined only by the final inliner, after
// the last __is_compile_time_constant resolution. `wrap` is big enough to avoid
// early inlining, so its __is_compile_time_constant(perm) must survive until
// inlining makes perm constant. Then every caller takes the const path, so the
// dynamic 99999 must not survive.

// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap -O2 --emit-llvm-text -o - \
// RUN:   | FileCheck %s --implicit-check-not="ret i32 99999"

// REQUIRES: X86_ENABLED

// CHECK: ret i32 1111

inline uniform int wrap(uniform int perm, uniform int buf[]) {
    uniform int acc = buf[0];
    acc ^= (acc*1000003) + buf[0]; buf[0] = acc;
    acc ^= (acc*1000004) + buf[1]; buf[1] = acc;
    acc ^= (acc*1000005) + buf[2]; buf[2] = acc;
    acc ^= (acc*1000006) + buf[3]; buf[3] = acc;
    acc ^= (acc*1000007) + buf[4]; buf[4] = acc;
    acc ^= (acc*1000008) + buf[5]; buf[5] = acc;
    acc ^= (acc*1000009) + buf[6]; buf[6] = acc;
    acc ^= (acc*1000010) + buf[7]; buf[7] = acc;
    acc ^= (acc*1000011) + buf[8]; buf[8] = acc;
    acc ^= (acc*1000012) + buf[9]; buf[9] = acc;
    acc ^= (acc*1000013) + buf[10]; buf[10] = acc;
    acc ^= (acc*1000014) + buf[11]; buf[11] = acc;
    acc ^= (acc*1000015) + buf[12]; buf[12] = acc;
    acc ^= (acc*1000016) + buf[13]; buf[13] = acc;
    acc ^= (acc*1000017) + buf[14]; buf[14] = acc;
    acc ^= (acc*1000018) + buf[15]; buf[15] = acc;
    acc ^= (acc*1000019) + buf[16]; buf[16] = acc;
    acc ^= (acc*1000020) + buf[17]; buf[17] = acc;
    acc ^= (acc*1000021) + buf[18]; buf[18] = acc;
    acc ^= (acc*1000022) + buf[19]; buf[19] = acc;
    acc ^= (acc*1000023) + buf[20]; buf[20] = acc;
    acc ^= (acc*1000024) + buf[21]; buf[21] = acc;
    acc ^= (acc*1000025) + buf[22]; buf[22] = acc;
    acc ^= (acc*1000026) + buf[23]; buf[23] = acc;
    acc ^= (acc*1000027) + buf[24]; buf[24] = acc;
    acc ^= (acc*1000028) + buf[25]; buf[25] = acc;
    acc ^= (acc*1000029) + buf[26]; buf[26] = acc;
    acc ^= (acc*1000030) + buf[27]; buf[27] = acc;
    acc ^= (acc*1000031) + buf[28]; buf[28] = acc;
    acc ^= (acc*1000032) + buf[29]; buf[29] = acc;
    acc ^= (acc*1000033) + buf[30]; buf[30] = acc;
    acc ^= (acc*1000034) + buf[31]; buf[31] = acc;
    acc ^= (acc*1000035) + buf[32]; buf[32] = acc;
    acc ^= (acc*1000036) + buf[33]; buf[33] = acc;
    acc ^= (acc*1000037) + buf[34]; buf[34] = acc;
    acc ^= (acc*1000038) + buf[35]; buf[35] = acc;
    acc ^= (acc*1000039) + buf[36]; buf[36] = acc;
    acc ^= (acc*1000040) + buf[37]; buf[37] = acc;
    acc ^= (acc*1000041) + buf[38]; buf[38] = acc;
    acc ^= (acc*1000042) + buf[39]; buf[39] = acc;
    acc ^= (acc*1000043) + buf[40]; buf[40] = acc;
    acc ^= (acc*1000044) + buf[41]; buf[41] = acc;
    acc ^= (acc*1000045) + buf[42]; buf[42] = acc;
    acc ^= (acc*1000046) + buf[43]; buf[43] = acc;
    acc ^= (acc*1000047) + buf[44]; buf[44] = acc;
    acc ^= (acc*1000048) + buf[45]; buf[45] = acc;
    acc ^= (acc*1000049) + buf[46]; buf[46] = acc;
    acc ^= (acc*1000050) + buf[47]; buf[47] = acc;
    acc ^= (acc*1000051) + buf[48]; buf[48] = acc;
    acc ^= (acc*1000052) + buf[49]; buf[49] = acc;
    acc ^= (acc*1000053) + buf[50]; buf[50] = acc;
    acc ^= (acc*1000054) + buf[51]; buf[51] = acc;
    acc ^= (acc*1000055) + buf[52]; buf[52] = acc;
    acc ^= (acc*1000056) + buf[53]; buf[53] = acc;
    acc ^= (acc*1000057) + buf[54]; buf[54] = acc;
    acc ^= (acc*1000058) + buf[55]; buf[55] = acc;
    acc ^= (acc*1000059) + buf[56]; buf[56] = acc;
    acc ^= (acc*1000060) + buf[57]; buf[57] = acc;
    acc ^= (acc*1000061) + buf[58]; buf[58] = acc;
    acc ^= (acc*1000062) + buf[59]; buf[59] = acc;
    acc ^= (acc*1000063) + buf[60]; buf[60] = acc;
    acc ^= (acc*1000064) + buf[61]; buf[61] = acc;
    acc ^= (acc*1000065) + buf[62]; buf[62] = acc;
    acc ^= (acc*1000066) + buf[63]; buf[63] = acc;
    acc ^= (acc*1000067) + buf[0]; buf[0] = acc;
    acc ^= (acc*1000068) + buf[1]; buf[1] = acc;
    acc ^= (acc*1000069) + buf[2]; buf[2] = acc;
    acc ^= (acc*1000070) + buf[3]; buf[3] = acc;
    acc ^= (acc*1000071) + buf[4]; buf[4] = acc;
    acc ^= (acc*1000072) + buf[5]; buf[5] = acc;
    acc ^= (acc*1000073) + buf[6]; buf[6] = acc;
    acc ^= (acc*1000074) + buf[7]; buf[7] = acc;
    acc ^= (acc*1000075) + buf[8]; buf[8] = acc;
    acc ^= (acc*1000076) + buf[9]; buf[9] = acc;
    acc ^= (acc*1000077) + buf[10]; buf[10] = acc;
    acc ^= (acc*1000078) + buf[11]; buf[11] = acc;
    acc ^= (acc*1000079) + buf[12]; buf[12] = acc;
    acc ^= (acc*1000080) + buf[13]; buf[13] = acc;
    acc ^= (acc*1000081) + buf[14]; buf[14] = acc;
    acc ^= (acc*1000082) + buf[15]; buf[15] = acc;
    acc ^= (acc*1000083) + buf[16]; buf[16] = acc;
    acc ^= (acc*1000084) + buf[17]; buf[17] = acc;
    acc ^= (acc*1000085) + buf[18]; buf[18] = acc;
    acc ^= (acc*1000086) + buf[19]; buf[19] = acc;
    acc ^= (acc*1000087) + buf[20]; buf[20] = acc;
    acc ^= (acc*1000088) + buf[21]; buf[21] = acc;
    acc ^= (acc*1000089) + buf[22]; buf[22] = acc;
    acc ^= (acc*1000090) + buf[23]; buf[23] = acc;
    acc ^= (acc*1000091) + buf[24]; buf[24] = acc;
    acc ^= (acc*1000092) + buf[25]; buf[25] = acc;
    acc ^= (acc*1000093) + buf[26]; buf[26] = acc;
    acc ^= (acc*1000094) + buf[27]; buf[27] = acc;
    acc ^= (acc*1000095) + buf[28]; buf[28] = acc;
    acc ^= (acc*1000096) + buf[29]; buf[29] = acc;
    acc ^= (acc*1000097) + buf[30]; buf[30] = acc;
    acc ^= (acc*1000098) + buf[31]; buf[31] = acc;
    acc ^= (acc*1000099) + buf[32]; buf[32] = acc;
    acc ^= (acc*1000100) + buf[33]; buf[33] = acc;
    acc ^= (acc*1000101) + buf[34]; buf[34] = acc;
    acc ^= (acc*1000102) + buf[35]; buf[35] = acc;
    acc ^= (acc*1000103) + buf[36]; buf[36] = acc;
    acc ^= (acc*1000104) + buf[37]; buf[37] = acc;
    acc ^= (acc*1000105) + buf[38]; buf[38] = acc;
    acc ^= (acc*1000106) + buf[39]; buf[39] = acc;
    acc ^= (acc*1000107) + buf[40]; buf[40] = acc;
    acc ^= (acc*1000108) + buf[41]; buf[41] = acc;
    acc ^= (acc*1000109) + buf[42]; buf[42] = acc;
    acc ^= (acc*1000110) + buf[43]; buf[43] = acc;
    acc ^= (acc*1000111) + buf[44]; buf[44] = acc;
    acc ^= (acc*1000112) + buf[45]; buf[45] = acc;
    acc ^= (acc*1000113) + buf[46]; buf[46] = acc;
    acc ^= (acc*1000114) + buf[47]; buf[47] = acc;
    acc ^= (acc*1000115) + buf[48]; buf[48] = acc;
    acc ^= (acc*1000116) + buf[49]; buf[49] = acc;
    acc ^= (acc*1000117) + buf[50]; buf[50] = acc;
    acc ^= (acc*1000118) + buf[51]; buf[51] = acc;
    acc ^= (acc*1000119) + buf[52]; buf[52] = acc;
    acc ^= (acc*1000120) + buf[53]; buf[53] = acc;
    acc ^= (acc*1000121) + buf[54]; buf[54] = acc;
    acc ^= (acc*1000122) + buf[55]; buf[55] = acc;
    acc ^= (acc*1000123) + buf[56]; buf[56] = acc;
    acc ^= (acc*1000124) + buf[57]; buf[57] = acc;
    acc ^= (acc*1000125) + buf[58]; buf[58] = acc;
    acc ^= (acc*1000126) + buf[59]; buf[59] = acc;
    acc ^= (acc*1000127) + buf[60]; buf[60] = acc;
    acc ^= (acc*1000128) + buf[61]; buf[61] = acc;
    acc ^= (acc*1000129) + buf[62]; buf[62] = acc;
    acc ^= (acc*1000130) + buf[63]; buf[63] = acc;
    acc ^= (acc*1000131) + buf[0]; buf[0] = acc;
    acc ^= (acc*1000132) + buf[1]; buf[1] = acc;
    acc ^= (acc*1000133) + buf[2]; buf[2] = acc;
    acc ^= (acc*1000134) + buf[3]; buf[3] = acc;
    acc ^= (acc*1000135) + buf[4]; buf[4] = acc;
    acc ^= (acc*1000136) + buf[5]; buf[5] = acc;
    acc ^= (acc*1000137) + buf[6]; buf[6] = acc;
    acc ^= (acc*1000138) + buf[7]; buf[7] = acc;
    acc ^= (acc*1000139) + buf[8]; buf[8] = acc;
    acc ^= (acc*1000140) + buf[9]; buf[9] = acc;
    acc ^= (acc*1000141) + buf[10]; buf[10] = acc;
    acc ^= (acc*1000142) + buf[11]; buf[11] = acc;
    acc ^= (acc*1000143) + buf[12]; buf[12] = acc;
    acc ^= (acc*1000144) + buf[13]; buf[13] = acc;
    acc ^= (acc*1000145) + buf[14]; buf[14] = acc;
    acc ^= (acc*1000146) + buf[15]; buf[15] = acc;
    acc ^= (acc*1000147) + buf[16]; buf[16] = acc;
    acc ^= (acc*1000148) + buf[17]; buf[17] = acc;
    acc ^= (acc*1000149) + buf[18]; buf[18] = acc;
    acc ^= (acc*1000150) + buf[19]; buf[19] = acc;
    acc ^= (acc*1000151) + buf[20]; buf[20] = acc;
    acc ^= (acc*1000152) + buf[21]; buf[21] = acc;
    acc ^= (acc*1000153) + buf[22]; buf[22] = acc;
    acc ^= (acc*1000154) + buf[23]; buf[23] = acc;
    acc ^= (acc*1000155) + buf[24]; buf[24] = acc;
    acc ^= (acc*1000156) + buf[25]; buf[25] = acc;
    acc ^= (acc*1000157) + buf[26]; buf[26] = acc;
    acc ^= (acc*1000158) + buf[27]; buf[27] = acc;
    acc ^= (acc*1000159) + buf[28]; buf[28] = acc;
    acc ^= (acc*1000160) + buf[29]; buf[29] = acc;
    acc ^= (acc*1000161) + buf[30]; buf[30] = acc;
    acc ^= (acc*1000162) + buf[31]; buf[31] = acc;
    acc ^= (acc*1000163) + buf[32]; buf[32] = acc;
    acc ^= (acc*1000164) + buf[33]; buf[33] = acc;
    acc ^= (acc*1000165) + buf[34]; buf[34] = acc;
    acc ^= (acc*1000166) + buf[35]; buf[35] = acc;
    acc ^= (acc*1000167) + buf[36]; buf[36] = acc;
    acc ^= (acc*1000168) + buf[37]; buf[37] = acc;
    acc ^= (acc*1000169) + buf[38]; buf[38] = acc;
    acc ^= (acc*1000170) + buf[39]; buf[39] = acc;
    acc ^= (acc*1000171) + buf[40]; buf[40] = acc;
    acc ^= (acc*1000172) + buf[41]; buf[41] = acc;
    acc ^= (acc*1000173) + buf[42]; buf[42] = acc;
    acc ^= (acc*1000174) + buf[43]; buf[43] = acc;
    acc ^= (acc*1000175) + buf[44]; buf[44] = acc;
    acc ^= (acc*1000176) + buf[45]; buf[45] = acc;
    acc ^= (acc*1000177) + buf[46]; buf[46] = acc;
    acc ^= (acc*1000178) + buf[47]; buf[47] = acc;
    acc ^= (acc*1000179) + buf[48]; buf[48] = acc;
    acc ^= (acc*1000180) + buf[49]; buf[49] = acc;
    acc ^= (acc*1000181) + buf[50]; buf[50] = acc;
    acc ^= (acc*1000182) + buf[51]; buf[51] = acc;
    acc ^= (acc*1000183) + buf[52]; buf[52] = acc;
    acc ^= (acc*1000184) + buf[53]; buf[53] = acc;
    acc ^= (acc*1000185) + buf[54]; buf[54] = acc;
    acc ^= (acc*1000186) + buf[55]; buf[55] = acc;
    acc ^= (acc*1000187) + buf[56]; buf[56] = acc;
    acc ^= (acc*1000188) + buf[57]; buf[57] = acc;
    acc ^= (acc*1000189) + buf[58]; buf[58] = acc;
    acc ^= (acc*1000190) + buf[59]; buf[59] = acc;
    acc ^= (acc*1000191) + buf[60]; buf[60] = acc;
    acc ^= (acc*1000192) + buf[61]; buf[61] = acc;
    acc ^= (acc*1000193) + buf[62]; buf[62] = acc;
    acc ^= (acc*1000194) + buf[63]; buf[63] = acc;
    acc ^= (acc*1000195) + buf[0]; buf[0] = acc;
    acc ^= (acc*1000196) + buf[1]; buf[1] = acc;
    acc ^= (acc*1000197) + buf[2]; buf[2] = acc;
    acc ^= (acc*1000198) + buf[3]; buf[3] = acc;
    acc ^= (acc*1000199) + buf[4]; buf[4] = acc;
    acc ^= (acc*1000200) + buf[5]; buf[5] = acc;
    acc ^= (acc*1000201) + buf[6]; buf[6] = acc;
    acc ^= (acc*1000202) + buf[7]; buf[7] = acc;
    buf[1] = acc;
    if (__is_compile_time_constant_uniform_int32(perm))
        return 11111 + perm; // const path
    return 99999;            // dynamic path (must be eliminated)
}
export uniform int call0(uniform int buf[]) { return wrap(3, buf); }
export uniform int call1(uniform int buf[]) { return wrap(4, buf); }
export uniform int call2(uniform int buf[]) { return wrap(5, buf); }
export uniform int call3(uniform int buf[]) { return wrap(6, buf); }
export uniform int call4(uniform int buf[]) { return wrap(7, buf); }
export uniform int call5(uniform int buf[]) { return wrap(8, buf); }
export uniform int call6(uniform int buf[]) { return wrap(9, buf); }
export uniform int call7(uniform int buf[]) { return wrap(10, buf); }
