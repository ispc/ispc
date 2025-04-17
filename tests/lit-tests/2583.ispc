// RUN: %{ispc} %s -o %t.o --target=host --nostdlib --nowrap 2>&1 | FileCheck %s --allow-empty

// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process
task void sum_M_float_task(uniform float v_out[], uniform float M[], uniform int M_pitch, uniform int N,
                      uniform int chunk_size) {
    uniform int col = taskIndex0 * chunk_size;

    uniform float *v_out_chunked = new float[chunk_size];

    foreach (row = 0 ... N) {
        v_out[row] = ((uniform float *)((uniform int8 *)M + col * M_pitch))[row];
    }
}
