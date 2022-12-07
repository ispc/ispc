// This test checks that DIArgList introduced in LLVM 13 does not appear for
// Xe targets when debug information is used.
// Related to llvm_patches/13_0_disable-DIArgList-in-SPIR-V.patch.
// Is only applicable to ISPC built with LLVM 13.0.

// RUN: %{ispc} %s --target=gen9-x8 -g --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_XE
// RUN: %{ispc} %s --target=avx2-i32x8 -g --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=CHECK_CPU

// REQUIRES: XE_ENABLED
// REQUIRES: LLVM_13_0+ && !LLVM_14_0+

struct Parameters
{
  int width;
  int height;
  int *output;
};
// CHECK_XE-NOT: call void @llvm.dbg.value(metadata !DIArgList
// CHECK_CPU: call void @llvm.dbg.value(metadata !DIArgList
task void test(void *uniform _p)
{
  Parameters *uniform p = (Parameters * uniform) _p;

  foreach (y = 0 ... p->height, x = 0 ... p->width) {
    int index        = y * p->width + x;
    p->output[index] = 1;
  }
}
