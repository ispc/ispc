// This test checks that the compiler reports an error (on `tile_q * tile_size`)
// in a foreach section and that this does not cause any crash (fatal error) 
// due to scoping issue when debugging is enabled.

// RUN: not %{ispc} --target=host --nowrap -g %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Can't convert from type "varying int32" to type "uniform int32" for initializer.
// CHECK-NOT: FATAL ERROR:

export void kernel(uniform int seq_len, uniform int tile_size) {
    uniform int num_tiles = (seq_len + tile_size - 1) / tile_size;
    foreach (tile_q = 0 ... num_tiles) {
        uniform int q_start = tile_q * tile_size;
        uniform int q_end = min(q_start + tile_size, seq_len);

        foreach (q_idx = q_start ... q_end) {
            float max_score = -1e20f;
        }
    }
}
