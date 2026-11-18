// Check AMX template immediates without the lowering pass and legacy calls with the pass enabled.
// RUN: %{ispc} %s --target=avx512gnr-x4 --nowrap --werror -O0 --off-phase=pre:last --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512gnr-x4 --nowrap --werror -DLEGACY_AMX -O2 --emit-llvm-text -o - | FileCheck %s
// RUN: not %{ispc} %s --target=avx512gnr-x4 --nowrap -DBAD_TILE=257 -O0 --off-phase=pre:last --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not %{ispc} %s --target=avx512gnr-x4 --nowrap -DBAD_TILE=-1 --opt=disable-assertions -O0 --off-phase=pre:last --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not %{ispc} %s --target=avx512gnr-x4 --nowrap -DBAD_TILE=4294967296ull --addressing=32 -O0 --off-phase=pre:last --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not %{ispc} %s --target=avx512gnr-x4 --nowrap -DBAD_DOT_TILE=257 -O0 --off-phase=pre:last --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not %{ispc} %s --target=avx512icl-x4 --nowrap -DUNSUPPORTED_AMX -O0 --off-phase=pre:last --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=NO-AMX
// RUN: not %{ispc} %s --target=avx512spr-x4 --nowrap -DUNSUPPORTED_FP16 -O0 --off-phase=pre:last --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=NO-FP16
// REQUIRES: X86_ENABLED && LLVM_20_0+ && !MACOS_HOST

#include <amx.isph>

#ifdef BAD_TILE
void check_range() { amx_tile_zero<BAD_TILE>(); }
// RANGE: Error: static assertion failed: AMX tile ID must be between 0 and 7.
#elif defined(BAD_DOT_TILE)
void check_dot_range() { amx_dpbssd<amx_tmm0, amx_tmm1, BAD_DOT_TILE>(); }
#elif defined(UNSUPPORTED_AMX)
void check_target() {
    amx_tile_zero<amx_tmm0>();
    amx_dpbf16ps<amx_tmm2, amx_tmm0, amx_tmm1>();
}
// NO-AMX-DAG: Error: Target specific LLVM intrinsic "llvm.x86.tilezero" requires feature "amxtile"
// NO-AMX-DAG: Error: Target specific LLVM intrinsic "llvm.x86.tdpbf16ps" requires feature "amxbf16"
#elif defined(UNSUPPORTED_FP16)
void check_fp16() { amx_dpfp16ps<amx_tmm2, amx_tmm0, amx_tmm1>(); }
// NO-FP16: Error: Target specific LLVM intrinsic "llvm.x86.tdpfp16ps" requires feature "amxfp16"
#else
constexpr uniform int accumulator_tile(uniform int first) { return first + 2; }

void check_immediates(const uniform int8 *uniform input, uniform int8 *uniform output, uniform int64 stride) {
#ifdef LEGACY_AMX
    amx_tile_zero(2);
    amx_tile_load(0, input, stride);
    amx_tile_load_t1(1, input, stride);
    amx_tile_store(2, output, stride);
    amx_dpbssd(2, 0, 1);
    amx_dpbsud(2, 0, 1);
    amx_dpbusd(2, 0, 1);
    amx_dpbuud(2, 0, 1);
    amx_dpbf16ps(2, 0, 1);
    amx_dpfp16ps(2, 0, 1);
#else
    constexpr uniform int accumulator = accumulator_tile(amx_tmm0);
    amx_tile_zero<accumulator>();
    amx_tile_load<amx_tmm0>(input, stride);
    amx_tile_load_t1<amx_tmm1>(input, stride);
    amx_tile_store<accumulator>(output, stride);
    amx_dpbssd<accumulator, amx_tmm0, amx_tmm1>();
    amx_dpbsud<accumulator, amx_tmm0, amx_tmm1>();
    amx_dpbusd<accumulator, amx_tmm0, amx_tmm1>();
    amx_dpbuud<accumulator, amx_tmm0, amx_tmm1>();
    amx_dpbf16ps<accumulator, amx_tmm0, amx_tmm1>();
    amx_dpfp16ps<accumulator, amx_tmm0, amx_tmm1>();
#endif
}

// CHECK-DAG: call void @llvm.x86.tilezero(i8 2)
// CHECK-DAG: call void @llvm.x86.tileloadd64(i8 0, ptr {{.*}}, i64 {{.*}})
// CHECK-DAG: call void @llvm.x86.tileloaddt164(i8 1, ptr {{.*}}, i64 {{.*}})
// CHECK-DAG: call void @llvm.x86.tilestored64(i8 2, ptr {{.*}}, i64 {{.*}})
// CHECK-DAG: call void @llvm.x86.tdpbssd(i8 2, i8 0, i8 1)
// CHECK-DAG: call void @llvm.x86.tdpbsud(i8 2, i8 0, i8 1)
// CHECK-DAG: call void @llvm.x86.tdpbusd(i8 2, i8 0, i8 1)
// CHECK-DAG: call void @llvm.x86.tdpbuud(i8 2, i8 0, i8 1)
// CHECK-DAG: call void @llvm.x86.tdpbf16ps(i8 2, i8 0, i8 1)
// CHECK-DAG: call void @llvm.x86.tdpfp16ps(i8 2, i8 0, i8 1)
#endif
