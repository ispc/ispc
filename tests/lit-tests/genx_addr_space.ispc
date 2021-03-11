// RUN: %{ispc} %s --target=genx-x16 --arch=genx64 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_GENX_X16 %s
// RUN: %{ispc} %s --target=genx-x16 --arch=genx32 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_GENX_X16 %s
// RUN: %{ispc} %s --target=genx-x8 --arch=genx64 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_GENX_X8 %s
// RUN: %{ispc} %s --target=genx-x8 --arch=genx32 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_GENX_X8 %s
// REQUIRES: GENX_ENABLED

struct vec3f {
    float x;
    float y;
    float z;
};

// CHECK_GENX_X16: call <16 x float> @llvm.genx.svm.block.ld.unaligned.v16f32.i64
// CHECK_GENX_X16: call void @llvm.genx.svm.block.st.i64.v16f32
// CHECK_GENX_X8: call <8 x float> @llvm.genx.svm.block.ld.unaligned.v8f32.i64
// CHECK_GENX_X8: call void @llvm.genx.svm.block.st.i64.v8f32
export void checkAddrSpace1(void *uniform _in, uniform float * uniform _out) {
  const varying vec3f *uniform in_test = (const varying vec3f *uniform) _in;
  _out[programIndex] = in_test->x;
}

// CHECK_GENX_X16: call <16 x float> @llvm.genx.svm.block.ld.unaligned.v16f32.i64
// CHECK_GENX_X16: call void @llvm.genx.svm.block.st.i64.v16f32
// CHECK_GENX_X8: call <8 x float> @llvm.genx.svm.block.ld.unaligned.v8f32.i64
// CHECK_GENX_X8: call void @llvm.genx.svm.block.st.i64.v8f32
export void checkAddrSpace2(void *uniform _out, uniform float * uniform _in) {
  varying vec3f *uniform out_test = (varying vec3f *uniform) _out;
  out_test->x = _in[programIndex];
}
