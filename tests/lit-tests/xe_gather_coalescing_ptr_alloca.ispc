// This test in conjunction with `xe_gather_coalescing_ptr_indvar.ispc` checks
//   for vectorization of consecitive loads to aggregate structures (`vec3f`).
// The tests are split to check for vectorization of directly-addressed,
//   alloca-allocated pointers and pointer-indirection to pointers based on
//   non-constant terms (indvars).


// This test checks desired behavior of gather coalescing optimization.
// Current implementation produces the following code sequence 9 times:
// %ptr209 = getelementptr float, float* %self_load3_color6_mem31.i184278, i64 %scaled_offset208
// %self_load = load float, float* %ptr209, align 4
// %self_load_insert = insertelement <8 x float> undef, float %self_load, i32 X
// %self_load_insert_shuffle = shufflevector <8 x float> %self_load_insert, <8 x float> undef, <8 x i32> zeroinitializer

// The optimization should optimize memory access by combining each 3 loads to one plus EEI:
// %vectorized_ld = call <3 x float> @llvm.genx.svm.block.ld.unaligned.v8f32.i64(i64 %vectorized_address)
// %vectorized_extr_elem223 = extractelement <3 x float> %vectorized_ld, i64 1
// %vectorized_extr_elem222 = extractelement <3 x float> %vectorized_ld, i64 2
// %vectorized_extr_elem221 = extractelement <3 x float> %vectorized_ld, i64 3

// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=321:321 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK_LOAD
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=321:321 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK_LOAD

// REQUIRES:  XE_ENABLED

struct vec3f {
    float x;
    float y;
    float z;
};

struct Data {
    uint8 *mem;
};

struct Linear {
    Data *opacity;
    Data *color;
};

static inline vec3f make_vec3f(vec3f a, vec3f b) {
    vec3f res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    res.z = a.z + b.z;
    return res;
}

static inline vec3f get_color(const uniform Linear *uniform self, int index) {
    return ((uniform vec3f * uniform) self->color->mem)[index];
}

#define NUM 64

// For all test cases below we expect the loads to the contiguous fields for the
// stack allocated vec3f structure (_in -> v0) to be coalesced.


// CHECK_ALL-LABEL: @gather_coalescing_i

// CHECK_LOAD:      %vectorized_ptrtoint = ptrtoint {{(float\*)|ptr}} %_in to i64
// CHECK_LOAD:      %vectorized_address = add i64 %vectorized_ptrtoint, 0
// CHECK_LOAD:      %vectorized_address_ptr = inttoptr i64 %vectorized_address to {{(<4 x float>\*)|ptr}}
// CHECK_LOAD:      %vectorized_ld_exp = load <4 x float>, {{(<4 x float>\*)|ptr}} %vectorized_address_ptr
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 0
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 1
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 2

task void gather_coalescing_i(uniform float _out[], uniform float _in[], uniform Linear *uniform _self) {
    vec3f v0 = {_in[0], _in[1], _in[2]};

    for (uniform int i = 0; i < NUM; i++)
        v0 = make_vec3f(v0, get_color(_self, i));

    _out[programIndex] = v0.x + v0.y + v0.z;
}

// CHECK_ALL-LABEL: @gather_coalescing_inogetter

// CHECK_LOAD:      %vectorized_ptrtoint = ptrtoint {{(float\*)|ptr}} %_in to i64
// CHECK_LOAD:      %vectorized_address = add i64 %vectorized_ptrtoint, 0
// CHECK_LOAD:      %vectorized_address_ptr = inttoptr i64 %vectorized_address to {{(<4 x float>\*)|ptr}}
// CHECK_LOAD:      %vectorized_ld_exp = load <4 x float>, {{(<4 x float>\*)|ptr}} %vectorized_address_ptr
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 0
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 1
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 2

task void gather_coalescing_inogetter(uniform float _out[], uniform float _in[], uniform Linear *uniform _self) {
    uniform uint8 color[1024];
    vec3f v0 = {_in[0], _in[1], _in[2]};

    for (uniform int i = 0; i < NUM; i++)
        v0 = make_vec3f(v0, ((uniform vec3f * uniform)&color)[i]);

    _out[programIndex] = v0.x + v0.y + v0.z;
}

// CHECK_ALL-LABEL: @gather_coalescing_ivarying

// CHECK_LOAD:      %vectorized_ptrtoint = ptrtoint {{(float\*)|ptr}} %_in to i64
// CHECK_LOAD:      %vectorized_address = add i64 %vectorized_ptrtoint, 0
// CHECK_LOAD:      %vectorized_address_ptr = inttoptr i64 %vectorized_address to {{(<4 x float>\*)|ptr}}
// CHECK_LOAD:      %vectorized_ld_exp = load <4 x float>, {{(<4 x float>\*)|ptr}} %vectorized_address_ptr
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 0
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 1
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 2

task void gather_coalescing_ivarying(uniform float _out[], uniform float _in[], uniform Linear *uniform _self) {
    vec3f v0 = {_in[0], _in[1], _in[2]};

    for (varying int i = programIndex; i < NUM; i += programCount)
        v0 = make_vec3f(v0, get_color(_self, i));

    _out[programIndex] = v0.x + v0.y + v0.z;
}

// CHECK_ALL-LABEL: @gather_coalescing_ivaryingnogetter

// CHECK_LOAD:      %vectorized_ptrtoint = ptrtoint {{(float\*)|ptr}} %_in to i64
// CHECK_LOAD:      %vectorized_address = add i64 %vectorized_ptrtoint, 0
// CHECK_LOAD:      %vectorized_address_ptr = inttoptr i64 %vectorized_address to {{(<4 x float>\*)|ptr}}
// CHECK_LOAD:      %vectorized_ld_exp = load <4 x float>, {{(<4 x float>\*)|ptr}} %vectorized_address_ptr
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 0
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 1
// CHECK_LOAD:      %mem_coal_eei{{[0-9]*}} = extractelement <4 x float> %vectorized_ld_exp, i64 2

task void gather_coalescing_ivaryingnogetter(uniform float _out[], uniform float _in[], uniform Linear *uniform _self) {
    uniform uint8 color[1024];
    vec3f v0 = {_in[0], _in[1], _in[2]};

    for (varying int i = programIndex; i < NUM; i += programCount)
        v0 = make_vec3f(v0, ((uniform vec3f * uniform)&color)[i]);

    _out[programIndex] = v0.x + v0.y + v0.z;
}
