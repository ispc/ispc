// This test in conjunction with `xe_gather_coalescing_ptr_alloca.ispc` checks
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
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK_LOAD,CHECK_VARYING
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=321:321 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK_LOAD,CHECK_VARYING

// REQUIRES: XE_ENABLED
// XFAIL:    XE_ENABLED


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


// For test cases with uniform induction variables, we would like the loads to
// the contiguous fields of the vec3f structure at index i to be coalesced.

// CHECK_ALL-LABEL: @gather_coalescing_i

// CHECK_LOAD-NOT:  %self_load{{[0-9]*}}_color{{[0-9]*}}_mem_{{.*}} = load float, float* %ptr{{[0-9]*}}
// CHECK_LOAD:      %vectorized_ptrtoint = ptrtoint float* %ptr{{[0-9]*}} to i64
// CHECK_LOAD:      %vectorized_address = add i64 %vectorized_ptrtoint, 0
// CHECK_LOAD:      %vectorized_address_ptr = inttoptr i64 %vectorized_address to <4 x float>*
// CHECK_LOAD:      %vectorized_ld_exp = load <4 x float>, <4 x float>* %vectorized_address_ptr
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

// CHECK_LOAD-NOT:  %color{{[0-9]*}}_{{.*}} = load float, float* %ptr{{[0-9]*}}
// CHECK_LOAD:      %vectorized_ptrtoint = ptrtoint float* %_in to i64
// CHECK_LOAD:      %vectorized_address = add i64 %vectorized_ptrtoint, 0
// CHECK_LOAD:      %vectorized_address_ptr = inttoptr i64 %vectorized_address to <4 x float>*
// CHECK_LOAD:      %vectorized_ld_exp = load <4 x float>, <4 x float>* %vectorized_address_ptr
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

// The tests cases below produce gather and scatter calls which currently
// cannot be coalesced.

// CHECK_ALL-LABEL: @gather_coalescing_ivarying

// CHECK_VARYING:   %res{{.*}} = {{(tail )?}}call <{{(8|16)}} x float> @llvm.genx.svm.gather.v{{(8|16)}}f32
// CHECK_VARYING:   %res{{.*}} = {{(tail )?}}call <{{(8|16)}} x float> @llvm.genx.svm.gather.v{{(8|16)}}f32
// CHECK_VARYING:   %res{{.*}} = {{(tail )?}}call <{{(8|16)}} x float> @llvm.genx.svm.gather.v{{(8|16)}}f32
// CHECK_VARYING:   call void @llvm.genx.svm.scatter.v{{(8|16)}}i1.v{{(8|16)}}i64.v{{(8|16)}}f32
// CHECK_VARYING:   call void @llvm.genx.svm.scatter.v{{(8|16)}}i1.v{{(8|16)}}i64.v{{(8|16)}}f32
// CHECK_VARYING:   call void @llvm.genx.svm.scatter.v{{(8|16)}}i1.v{{(8|16)}}i64.v{{(8|16)}}f32

task void gather_coalescing_ivarying(uniform float _out[], uniform float _in[], uniform Linear *uniform _self) {
    vec3f v0 = {_in[0], _in[1], _in[2]};

    for (varying int i = programIndex; i < NUM; i += programCount)
        v0 = make_vec3f(v0, get_color(_self, i));

    _out[programIndex] = v0.x + v0.y + v0.z;
}

// CHECK_ALL-LABEL: @gather_coalescing_ivaryingnogetter

// CHECK_VARYING:   %res{{.*}} = {{(tail )?}}call <{{(8|16)}} x float> @llvm.genx.svm.gather.v{{(8|16)}}f32
// CHECK_VARYING:   %res{{.*}} = {{(tail )?}}call <{{(8|16)}} x float> @llvm.genx.svm.gather.v{{(8|16)}}f32
// CHECK_VARYING:   %res{{.*}} = {{(tail )?}}call <{{(8|16)}} x float> @llvm.genx.svm.gather.v{{(8|16)}}f32
// CHECK_VARYING:   call void @llvm.genx.svm.scatter.v{{(8|16)}}i1.v{{(8|16)}}i64.v{{(8|16)}}f32
// CHECK_VARYING:   call void @llvm.genx.svm.scatter.v{{(8|16)}}i1.v{{(8|16)}}i64.v{{(8|16)}}f32
// CHECK_VARYING:   call void @llvm.genx.svm.scatter.v{{(8|16)}}i1.v{{(8|16)}}i64.v{{(8|16)}}f32

task void gather_coalescing_ivaryingnogetter(uniform float _out[], uniform float _in[], uniform Linear *uniform _self) {
    uniform uint8 color[1024];
    vec3f v0 = {_in[0], _in[1], _in[2]};

    for (varying int i = programIndex; i < NUM; i += programCount)
        v0 = make_vec3f(v0, ((uniform vec3f * uniform)&color)[i]);

    _out[programIndex] = v0.x + v0.y + v0.z;
}
