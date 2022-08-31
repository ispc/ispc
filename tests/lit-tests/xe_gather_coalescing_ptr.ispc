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

// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

// REQUIRES: XE_ENABLED
// XFAIL: XE_ENABLED

// CHECK: %vectorized_ld = call <3 x float> @llvm.genx.svm.block.ld

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
task void gather_coalescing(uniform float _out[], uniform float _in[], uniform Linear *uniform _self) {
    vec3f v0 = {_in[0], _in[1], _in[2]};
    for (uniform int i = 0; i < NUM; i++) {
        v0 = make_vec3f(v0, get_color(_self, i & 1));
        v0 = make_vec3f(v0, get_color(_self, i & 3));
        v0 = make_vec3f(v0, get_color(_self, i & 5));
    }

    _out[programIndex] = v0.x + v0.y + v0.z;
}
