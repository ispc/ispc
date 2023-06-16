// This test case checks for more difficult block load coalescing opportunities.
// After loop unrolling or inlining multiple loads to contiguous addresses,
//   the XeGatherCoalescing pass should be able to coalesce multiple loads into
//   fewer block loads.
// However, the induction variables and non-constant terms used for the pointer
//   arithmetic here causes the XeGatherCoalescing pass to perform
//   no optimization.


// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=321:321 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK_LOAD_PRAGMA4,CHECK_LOAD_INLINE4,CHECK_LOAD_INLINE8
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=321:321 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK_LOAD_PRAGMA4,CHECK_LOAD_INLINE4,CHECK_LOAD_INLINE8

// REQUIRES: XE_ENABLED
// XFAIL:    XE_ENABLED

#define LOAD(n)   \
    a[n] = _in[n]

#define NUM 64


// CHECK_ALL-LABEL:     @gather_coalescing_loopunroll4pragma

// CHECK_LOAD_PRAGMA4:  %vectorized_ptrtoint = ptrtoint float* %_in to i64
// CHECK_LOAD_PRAGMA4:  %vectorized_address = add i64 %vectorized_ptrtoint, 0
// CHECK_LOAD_PRAGMA4:  %vectorized_address_ptr = inttoptr i64 %vectorized_address to <4 x float>*
// CHECK_LOAD_PRAGMA4:  %vectorized_ld_exp = load <4 x float>, <4 x float>* %vectorized_address_ptr
// CHECK_LOAD_PRAGMA4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 0
// CHECK_LOAD_PRAGMA4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 1
// CHECK_LOAD_PRAGMA4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 2
// CHECK_LOAD_PRAGMA4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 3

task void gather_coalescing_loopunroll4pragma(uniform float _out[], uniform float _in[]) {
    uniform float a[NUM];

    // Initialization
    for (uniform int i = 0; i < NUM; i++)
        a[i] = 0.0f;

#pragma unroll 4
    for (uniform int i = 0; i < NUM; i++)
        LOAD(i);

    // Perform calculation on loaded values
    for (uniform int i = 0; i < NUM; i++)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}


// CHECK_ALL-LABEL:     @gather_coalescing_loopunroll4inline

// CHECK_LOAD_INLINE4:  %vectorized_ptrtoint = ptrtoint float* %_in to i64
// CHECK_LOAD_INLINE4:  %vectorized_address = add i64 %vectorized_ptrtoint, 0
// CHECK_LOAD_INLINE4:  %vectorized_address_ptr = inttoptr i64 %vectorized_address to <4 x float>*
// CHECK_LOAD_INLINE4:  %vectorized_ld_exp = load <4 x float>, <4 x float>* %vectorized_address_ptr
// CHECK_LOAD_INLINE4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 0
// CHECK_LOAD_INLINE4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 1
// CHECK_LOAD_INLINE4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 2
// CHECK_LOAD_INLINE4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 3

task void gather_coalescing_loopunroll4inline(uniform float _out[], uniform float _in[]) {
    uniform float a[NUM];

    // Initialization
    for (uniform int i = 0; i < NUM; ++i)
        a[i] = 0.0f;

    for (uniform int i = 0; i < NUM; i += 4) {
        if (i + 3 < NUM) {
            LOAD(i);
            LOAD(i + 1);
            LOAD(i + 2);
            LOAD(i + 3);
        }
    }

    // Perform calculation on loaded values
    for (uniform int i = 0; i < NUM; i++)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}


// CHECK_ALL-LABEL:     @gather_coalescing_loopunroll8inline

// CHECK_LOAD_INLINE8:  %vectorized_ptrtoint = ptrtoint float* %_in to i64
// CHECK_LOAD_INLINE8:  %vectorized_address = add i64 %vectorized_ptrtoint, 0
// CHECK_LOAD_INLINE8:  %vectorized_address_ptr = inttoptr i64 %vectorized_address to <8 x float>*
// CHECK_LOAD_INLINE8:  %vectorized_ld_exp = load <8 x float>, <8 x float>* %vectorized_address_ptr
// CHECK_LOAD_INLINE8:  %mem_coal_eei{{[0-9]*}} = extractelement <8 x float> %vectorized_ld_exp, i64 0
// CHECK_LOAD_INLINE8:  %mem_coal_eei{{[0-9]*}} = extractelement <8 x float> %vectorized_ld_exp, i64 1
// CHECK_LOAD_INLINE8:  %mem_coal_eei{{[0-9]*}} = extractelement <8 x float> %vectorized_ld_exp, i64 2
// CHECK_LOAD_INLINE8:  %mem_coal_eei{{[0-9]*}} = extractelement <8 x float> %vectorized_ld_exp, i64 3
// CHECK_LOAD_INLINE8:  %mem_coal_eei{{[0-9]*}} = extractelement <8 x float> %vectorized_ld_exp, i64 4
// CHECK_LOAD_INLINE8:  %mem_coal_eei{{[0-9]*}} = extractelement <8 x float> %vectorized_ld_exp, i64 5
// CHECK_LOAD_INLINE8:  %mem_coal_eei{{[0-9]*}} = extractelement <8 x float> %vectorized_ld_exp, i64 6
// CHECK_LOAD_INLINE8:  %mem_coal_eei{{[0-9]*}} = extractelement <8 x float> %vectorized_ld_exp, i64 7

task void gather_coalescing_loopunroll8inline(uniform float _out[], uniform float _in[]) {
    uniform float a[NUM];

    // Initialization
    for (uniform int i = 0; i < NUM; i++)
        a[i] = 0.0f;

    for (uniform int i = 0; i < NUM; i += 8) {
        if (i + 7 < NUM) {
            LOAD(i);
            LOAD(i + 1);
            LOAD(i + 2);
            LOAD(i + 3);
            LOAD(i + 4);
            LOAD(i + 5);
            LOAD(i + 6);
            LOAD(i + 7);
        }
    }

    // Perform calculation on loaded values
    for (uniform int i = 0; i < NUM; i++)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}
