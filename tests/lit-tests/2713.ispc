// RUN: %{ispc} --target=neon-i32x8 %s -o %t.o 2>&1 | FileCheck %s

// LLVM 16 requires backported patch to disable loop alignment.

// REQUIRES: ARM_ENABLED
// REQUIRES: LLVM_16_0+

// CHECK-NOT: LLVM ERROR: Failed to evaluate function length in SEH unwind info

export void resample(uniform uint32 width, uniform uint32 height,
                    uniform const uint8 src[], uniform uint8 out[]) {
    foreach_tiled (y = 0 ... height, x = 0 ... width) {
        out[x] = src[y];
    }
}
