// This test ensures that the target suffix is appended to the output file name.

// RUN: %{ispc} %s -h %t.h --nostdlib --target=sse4-i32x4,avx2-i32x8,avx512skx-x16 -o %t.obj
// RUN: ls %t_sse4.obj %t_avx2.obj %t_avx512skx.obj
// RUN: %{ispc} %s -h %t.h --nostdlib --target=sse4-i32x4,avx2-i32x8,avx512skx-x16 -o ispc-obj
// RUN: ls ispc-obj_sse4 ispc-obj_avx2 ispc-obj_avx512skx

// REQUIRES: !WINDOWS_ENABLED && X86_ENABLED

#define NUM 1024

export void foo(uniform int * uniform _in, uniform int * uniform _out, uniform int k)
{
    uniform int array[NUM];
    for (uniform int i = 0; i < NUM; i++) {
        array[i] = _in[i] * k;
    }
    _out[programIndex] = array[programIndex];
}
