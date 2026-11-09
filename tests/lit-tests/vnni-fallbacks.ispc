// Checks the code emitted for the dot4add_*/dot2add_* functions on targets without the matching VNNI instructions.
// The fallbacks must stay vectorized (no scalar imul) and must not use VNNI forms the target lacks. On avx2vnni,
// the forms with native support use the VEX encoding. On the VNNI targets, the 8-bit forms without native support
// are built from exactly two vpdpbusd per vector (V1, V2 and V4 give the vectors per gang) and
// dot2add_u16i16packed from vpdpwssd.

// RUN: %{ispc} %s --target=sse4.2-i32x4 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,NOVNNI --implicit-check-not=imul \
// RUN:   --implicit-check-not=vpdp
// RUN: %{ispc} %s --target=avx1-i32x8 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,NOVNNI --implicit-check-not=imul \
// RUN:   --implicit-check-not=vpdp
// RUN: %{ispc} %s --target=avx2-i32x8 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,NOVNNI --implicit-check-not=imul \
// RUN:   --implicit-check-not=vpdp
// RUN: %{ispc} %s --target=avx512skx-x16 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,NOVNNI --implicit-check-not=imul \
// RUN:   --implicit-check-not=vpdp

// RUN: %{ispc} %s --target=avx2vnni-i32x4 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVXVNNI,VNNI,V1 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx2vnni-i32x8 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVXVNNI,VNNI,V1 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx2vnni-i32x16 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVXVNNI,VNNI,V2 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'

// RUN: %{ispc} %s --target=avx512icl-x16 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVX512VNNI,VNNI,V1 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512spr-x16 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVX512VNNI,VNNI,V1 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512gnr-x16 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVX512VNNI,VNNI,V1 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512icl-x4 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVX512VNNI,VNNI,V1 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512icl-x8 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVX512VNNI,VNNI,V1 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512icl-x32 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVX512VNNI,VNNI,V2 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512icl-x64 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVX512VNNI,VNNI,V4 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512icl-x16 --opt=disable-zmm --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVX512VNNI,VNNI,V2 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// SPR and GNR also have AVX-VNNI, so at 128- and 256-bit widths LLVM uses the VEX encoding.
// RUN: %{ispc} %s --target=avx512spr-x4 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVXVNNI,VNNI,V1 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512spr-x8 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVXVNNI,VNNI,V1 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512gnr-x4 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVXVNNI,VNNI,V1 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512gnr-x8 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVXVNNI,VNNI,V1 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512spr-x32 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVX512VNNI,VNNI,V2 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'
// RUN: %{ispc} %s --target=avx512spr-x64 --target-os=linux --emit-asm -o - \
// RUN:   | FileCheck %s -check-prefixes=CHECK,AVX512VNNI,VNNI,V4 --implicit-check-not=imul \
// RUN:   --implicit-check-not='{{vpdpb(ss|su|uu)|vpdpw(su|us|uu)}}'

// At -O0, the paths for instructions the target lacks must still be removed: no calls to the __not_supported
// stubs or to dot builtins.
// RUN: %{ispc} %s --target=avx2-i32x8 --target-os=linux -O0 --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=O0 --implicit-check-not=__not_supported \
// RUN:   '--implicit-check-not={{__dot[24]add_}}'
// RUN: %{ispc} %s --target=avx2vnni-i32x8 --target-os=linux -O0 --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=O0 --implicit-check-not=__not_supported \
// RUN:   '--implicit-check-not={{__dot[24]add_}}'
// RUN: %{ispc} %s --target=avx512icl-x16 --target-os=linux -O0 --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=O0 --implicit-check-not=__not_supported \
// RUN:   '--implicit-check-not={{__dot[24]add_}}'

// REQUIRES: X86_ENABLED && LINUX_ENABLED

// O0: {{^}}dot4_u8i8___{{.*}}:

// --implicit-check-not rejects scalar multiplies everywhere, any VNNI instruction on NOVNNI targets, and on the
// VNNI targets the AVX-VNNI-INT8/INT16 forms (vpdpbssd, vpdpbsud, vpdpbuud, vpdpwsud, vpdpwusd, vpdpwuud and their
// saturating variants), which none of them has.

// CHECK-LABEL: {{^}}dot4_u8i8___{{.*}}:
// NOVNNI: {{v?}}pmaddwd
// AVXVNNI: {vex} vpdpbusd{{[ \t]}}
// AVX512VNNI: {{^[ \t]+}}vpdpbusd{{[ \t]}}
// CHECK: ret
unmasked void dot4_u8i8(uniform uint a[], uniform uint b[], uniform int acc[], uniform int dst[]) {
    dst[programIndex] = dot4add_u8i8packed(a[programIndex], b[programIndex], acc[programIndex]);
}

// CHECK-LABEL: {{^}}dot4_u8i8_sat___{{.*}}:
// NOVNNI: {{v?}}pmaddwd
// AVXVNNI: {vex} vpdpbusds{{[ \t]}}
// AVX512VNNI: {{^[ \t]+}}vpdpbusds{{[ \t]}}
// CHECK: ret
unmasked void dot4_u8i8_sat(uniform uint a[], uniform uint b[], uniform int acc[], uniform int dst[]) {
    dst[programIndex] = dot4add_u8i8packed_sat(a[programIndex], b[programIndex], acc[programIndex]);
}

// CHECK-LABEL: {{^}}dot4_u8u8___{{.*}}:
// NOVNNI: {{v?}}pmaddwd
// V1-COUNT-2: vpdpbusd{{[ \t]}}
// V2-COUNT-4: vpdpbusd{{[ \t]}}
// V4-COUNT-8: vpdpbusd{{[ \t]}}
// VNNI-NOT: vpdpbusd
// CHECK: ret
unmasked void dot4_u8u8(uniform uint a[], uniform uint b[], uniform uint acc[], uniform uint dst[]) {
    dst[programIndex] = dot4add_u8u8packed(a[programIndex], b[programIndex], acc[programIndex]);
}

// CHECK-LABEL: {{^}}dot4_u8u8_sat___{{.*}}:
// NOVNNI: {{v?}}pmaddwd
// V1-COUNT-2: vpdpbusd{{[ \t]}}
// V2-COUNT-4: vpdpbusd{{[ \t]}}
// V4-COUNT-8: vpdpbusd{{[ \t]}}
// VNNI-NOT: vpdpbusd
// CHECK: ret
unmasked void dot4_u8u8_sat(uniform uint a[], uniform uint b[], uniform uint acc[], uniform uint dst[]) {
    dst[programIndex] = dot4add_u8u8packed_sat(a[programIndex], b[programIndex], acc[programIndex]);
}

// CHECK-LABEL: {{^}}dot4_i8i8___{{.*}}:
// NOVNNI: {{v?}}p{{(mulld|maddwd|mullw)}}
// V1-COUNT-2: vpdpbusd{{[ \t]}}
// V2-COUNT-4: vpdpbusd{{[ \t]}}
// V4-COUNT-8: vpdpbusd{{[ \t]}}
// VNNI-NOT: vpdpbusd
// CHECK: ret
unmasked void dot4_i8i8(uniform uint a[], uniform uint b[], uniform int acc[], uniform int dst[]) {
    dst[programIndex] = dot4add_i8i8packed(a[programIndex], b[programIndex], acc[programIndex]);
}

// CHECK-LABEL: {{^}}dot4_i8i8_sat___{{.*}}:
// NOVNNI: {{v?}}p{{(mulld|maddwd|mullw)}}
// V1-COUNT-2: vpdpbusd{{[ \t]}}
// V2-COUNT-4: vpdpbusd{{[ \t]}}
// V4-COUNT-8: vpdpbusd{{[ \t]}}
// VNNI-NOT: vpdpbusd
// CHECK: ret
unmasked void dot4_i8i8_sat(uniform uint a[], uniform uint b[], uniform int acc[], uniform int dst[]) {
    dst[programIndex] = dot4add_i8i8packed_sat(a[programIndex], b[programIndex], acc[programIndex]);
}

// CHECK-LABEL: {{^}}dot2_i16i16___{{.*}}:
// NOVNNI: {{v?}}pmaddwd
// AVXVNNI: {vex} vpdpwssd{{[ \t]}}
// AVX512VNNI: {{^[ \t]+}}vpdpwssd{{[ \t]}}
// CHECK: ret
unmasked void dot2_i16i16(uniform uint a[], uniform uint b[], uniform int acc[], uniform int dst[]) {
    dst[programIndex] = dot2add_i16i16packed(a[programIndex], b[programIndex], acc[programIndex]);
}

// CHECK-LABEL: {{^}}dot2_i16i16_sat___{{.*}}:
// NOVNNI: {{v?}}pmaddwd
// AVXVNNI: {vex} vpdpwssds{{[ \t]}}
// AVX512VNNI: {{^[ \t]+}}vpdpwssds{{[ \t]}}
// CHECK: ret
unmasked void dot2_i16i16_sat(uniform uint a[], uniform uint b[], uniform int acc[], uniform int dst[]) {
    dst[programIndex] = dot2add_i16i16packed_sat(a[programIndex], b[programIndex], acc[programIndex]);
}

// CHECK-LABEL: {{^}}dot2_u16u16___{{.*}}:
// CHECK: {{v?}}p{{(mulld|maddwd|mullw)}}
// CHECK: ret
unmasked void dot2_u16u16(uniform uint a[], uniform uint b[], uniform uint acc[], uniform uint dst[]) {
    dst[programIndex] = dot2add_u16u16packed(a[programIndex], b[programIndex], acc[programIndex]);
}

// CHECK-LABEL: {{^}}dot2_u16u16_sat___{{.*}}:
// CHECK: {{v?}}p{{(mulld|maddwd|mullw)}}
// CHECK: ret
unmasked void dot2_u16u16_sat(uniform uint a[], uniform uint b[], uniform uint acc[], uniform uint dst[]) {
    dst[programIndex] = dot2add_u16u16packed_sat(a[programIndex], b[programIndex], acc[programIndex]);
}

// CHECK-LABEL: {{^}}dot2_u16i16___{{.*}}:
// NOVNNI: {{v?}}p{{(mulld|maddwd|mullw)}}
// Two vpdpwssd per vector, with acc = 0; LLVM may turn some of them into vpmaddwd.
// V1-COUNT-2: {{vpdpwssd|vpmaddwd}}
// V2-COUNT-4: {{vpdpwssd|vpmaddwd}}
// V4-COUNT-8: {{vpdpwssd|vpmaddwd}}
// VNNI-NOT: {{vpdpwssd|vpmaddwd}}
// CHECK: ret
unmasked void dot2_u16i16(uniform uint a[], uniform uint b[], uniform int acc[], uniform int dst[]) {
    dst[programIndex] = dot2add_u16i16packed(a[programIndex], b[programIndex], acc[programIndex]);
}

// CHECK-LABEL: {{^}}dot2_u16i16_sat___{{.*}}:
// CHECK: {{v?}}p{{(mulld|maddwd|mullw)}}
// CHECK: ret
unmasked void dot2_u16i16_sat(uniform uint a[], uniform uint b[], uniform int acc[], uniform int dst[]) {
    dst[programIndex] = dot2add_u16i16packed_sat(a[programIndex], b[programIndex], acc[programIndex]);
}
