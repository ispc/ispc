// RUN: %{ispc} %s --target=genx --arch=genx64 --addressing=64 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// RUN: %{ispc} %s --target=genx --arch=genx32 --addressing=32 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// REQUIRES: GENX_ENABLED

struct vec3f {
    float x;
    float y;
    float z;
};

// CHECK: alloca [3 x %v16_varying_vec3f]
// CHECK: load float, float* %_in
// CHECK: load <16 x float>, <16 x float>* %ptr_cast_for_load
// CHECK: call void @llvm.genx.svm.block.st.v16f32(i64 %svm_st_ptrtoint
export void checkAddrSpace(uniform float _out[], uniform float _in[]) {
    float a = _in[programIndex];
    vec3f myFoo[3] = { { -1, -2, -3 }, {a, -3, 0}, {-4, -5, -6} };
    int i = _in[0];
    vec3f barFoo = myFoo[i];
    _out[programIndex] = barFoo.x;
}
