// This test case checks for the XeGatherCoalescing pass's ability to coalesce
//   block loads in the simple case - inline loads to contiguous addresses.
// This test should pass more easily relative to cases where the block loads
//   occur inside a loop or uses non-constant terms for indexing - these cases
//   are split into separate tests.


// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=321:321 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK_LOAD_INLINE4,CHECK_LOAD_INLINE16
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=321:321 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK_LOAD_INLINE4,CHECK_LOAD_INLINE16

// REQUIRES:  XE_ENABLED

#define LOAD(n)   \
    a[n] = _in[n]

#define NUM 64


// CHECK_ALL-LABEL:     @gather_coalescing_inline4

// CHECK_LOAD_INLINE4:  %vectorized_ptrtoint = ptrtoint {{(float\*)|ptr}} %_in to i64
// CHECK_LOAD_INLINE4:  %vectorized_address = add i64 %vectorized_ptrtoint, 0
// CHECK_LOAD_INLINE4:  %vectorized_address_ptr = inttoptr i64 %vectorized_address to {{(<4 x float>\*)|ptr}}
// CHECK_LOAD_INLINE4:  %vectorized_ld_exp = load <4 x float>, {{(<4 x float>\*)|ptr}} %vectorized_address_ptr
// CHECK_LOAD_INLINE4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 0
// CHECK_LOAD_INLINE4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 1
// CHECK_LOAD_INLINE4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 2
// CHECK_LOAD_INLINE4:  %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 3

task void gather_coalescing_inline4(uniform float _out[], uniform float _in[]) {
    uniform float a[NUM];

    // Initialization
    for (uniform int i = 0; i < NUM; i++)
        a[i] = 0.0f;

    LOAD(0);
    LOAD(1);
    LOAD(2);
    LOAD(3);

    // Perform calculation on loaded values
    for (uniform int i = 0; i < NUM; i++)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}

// CHECK_ALL-LABEL:       @gather_coalescing_inline16

// CHECK_LOAD_INLINE16:   %vectorized_ptrtoint = ptrtoint {{(float\*)|ptr}} %_in to i64
// CHECK_LOAD_INLINE16:   %vectorized_address = add i64 %vectorized_ptrtoint, 0
// CHECK_LOAD_INLINE16:   %vectorized_address_ptr = inttoptr i64 %vectorized_address to {{(<16 x float>\*)|ptr}}
// CHECK_LOAD_INLINE16:   %vectorized_ld_exp = load <16 x float>, {{(<16 x float>\*)|ptr}} %vectorized_address_ptr
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 0
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 1
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 2
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 3
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 4
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 5
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 6
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 7
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 8
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 9
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 10
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 11
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 12
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 13
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 14
// CHECK_LOAD_INLINE16:   %mem_coal_eei{{[0-9]*}} = extractelement <16 x float> %vectorized_ld_exp, i64 15

task void gather_coalescing_inline16(uniform float _out[], uniform float _in[]) {
    uniform float a[NUM];

    // Initialization
    for (uniform int i = 0; i < NUM; i++)
        a[i] = 0.0f;

    LOAD(0);
    LOAD(1);
    LOAD(2);
    LOAD(3);
    LOAD(4);
    LOAD(5);
    LOAD(6);
    LOAD(7);
    LOAD(8);
    LOAD(9);
    LOAD(10);
    LOAD(11);
    LOAD(12);
    LOAD(13);
    LOAD(14);
    LOAD(15);

    // Perform calculation on loaded values
    for (uniform int i = 0; i < NUM; i++)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}
