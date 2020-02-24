// RUN: %{ispc} %s --target=genx --arch=genx64 --addressing=64 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// RUN: %{ispc} %s --target=genx --arch=genx32 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// REQUIRES: GENX_ENABLED

struct vec3f {
    float x;
    float y;
    float z;
};

// CHECK: call <16 x float> @llvm.genx.svm.block.ld.v16f32
// CHECK: call void @llvm.genx.svm.block.st.v16f32
export void checkAddrSpace1(void *uniform _in, uniform float * uniform _out) {
  const varying vec3f *uniform in_test = (const varying vec3f *uniform) _in;
  _out[programIndex] = in_test->x;
}

// CHECK: call <16 x float> @llvm.genx.svm.block.ld.v16f32
// CHECK: call void @llvm.genx.svm.block.st.v16f32
export void checkAddrSpace2(void *uniform _out, uniform float * uniform _in) {
  varying vec3f *uniform out_test = (varying vec3f *uniform) _out;
  out_test->x = _in[programIndex];
}
