// RUN: %{ispc} %s --target=sse2-i32x4 -o %t.o
// RUN: %{ispc} %s --target=sse2-i32x4 --emit-llvm-text --debug-phase=261:261 -o %t.o 2>&1 | FileCheck  %s --check-prefixes CHECK

// REQUIRES: X86_ENABLED

// CHECK: %first_elt = extractelement <4 x i32> %__mask, i32 0

struct vec3ui { unsigned int x, y, z; };

static const uniform size_t N = 64;

export void VdbSampler_computeSampleM_stream4() {
  uniform vec3ui domainOffset[4 * 64];
  if (lanemask() == 15) {
    uniform uint32 activeInstance;
    if (reduce_equal(programIndex, &activeInstance)) {
      foreach (o = 0 ... N) {
        domainOffset[o + activeInstance];
      }
    }
  }
}
