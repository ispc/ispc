// RUN: %{ispc} %s --target=avx512skx-x8,avx2-i32x8 --nostdlib -o %t.o
// RUN: %{ispc} %s --target=avx2-i32x8,avx512skx-x8 --nostdlib -o %t.o
// RUN: %{ispc} %s --target=avx512skx-x16,avx2-i32x16 --nostdlib -o %t.o
// RUN: %{ispc} %s --target=avx512skx-x4,avx2-i32x4 --nostdlib -o %t.o

// REQUIRES: X86_ENABLED

// Struct cache should be reset between compilation of different targets in multi-target compialtion.
// If it's not done, in this test case it will cause a compiler fail, if targets of the same width
// have different mask definition (i.e. <i1 x 8> vs <i32 x 8>).

struct Sampler {
  varying float (*uniform computeSample_varying)(const Sampler *uniform _self);
};

varying float computeSample(const Sampler *uniform self)
{
  return 1.f;
}

export void func_call_with_ptr(void *uniform _sampler)
{
  Sampler *uniform sampler = (Sampler *uniform) _sampler;
  sampler->computeSample_varying = computeSample;
}
