//; RUN: %{ispc} %s --target=avx2-i32x4    -o %t.o --nostdlib
//; RUN: %{ispc} %s --target=avx512knl-x16 -o %t.o --nostdlib
//; RUN: %{ispc} %s --target=avx512skx-x4  -o %t.o --nostdlib
//; RUN: %{ispc} %s --target=neon-i32x4    -o %t.o --nostdlib

//; RUN: not %{ispc} %s --target=sse2-i32x4 -o %t.o --nostdlib
//; RUN: not %{ispc} %s --target=sse4-i32x4 -o %t.o --nostdlib
//; RUN: not %{ispc} %s --target=avx1-i32x4 -o %t.o --nostdlib

// REQUIRES: X86_ENABLED

// Make sure it compiles and doesn't fail during const folding in the front-end
float16 f11() {
  return (programIndex % 2) ? 1.23f16 : 4.56f16;
}
