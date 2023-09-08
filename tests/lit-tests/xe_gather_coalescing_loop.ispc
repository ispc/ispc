// This test case checks for the simple load in a loop case.
// Since no unrolling occurs in this case, there is no coalescing
//   for the XeGatherCoalescing pass to perform.


// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=321:321 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=321:321 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK

// REQUIRES:  XE_ENABLED

#define LOAD(n)     \
    a[n] = _in[n]

#define NUM 64


// CHECK_ALL-LABEL: @gather_coalescing_loop

// CHECK:           %_in_load_offset = getelementptr float, {{(float\*)|ptr}} %_in
// CHECK:           %_in_load_offset_load = load float, {{(float\*)|ptr}} %_in_load_offset

task void gather_coalescing_loop(uniform float _out[], uniform float _in[]) {
    uniform float a[NUM];

    // Initialization
    for (uniform int i = 0; i < NUM; i++)
        a[i] = 0.0f;

    for (uniform int i = 0; i < NUM; i++)
        LOAD(i);

    // Perform calculation on loaded values
    for (uniform int i = 0; i < NUM; i++)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}
