// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe32 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe32 -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s
// REQUIRES: XE_ENABLED

#define LOAD(n)               \
    do {                      \
        if (n < programCount) \
            a[n] = _in[n];   \
    } while (false)

// CHECK: @llvm.genx.svm.block.ld
export void gather_coalescing(uniform float _out[], uniform float _in[]) {
    uniform int a[programCount];

    // Initialization
    for (uniform int i = 0; i < programCount; ++i)
        a[i] = 0;

    LOAD(0);
    LOAD(1);
    LOAD(2);
    LOAD(3);
    LOAD(4);
    LOAD(5);
    LOAD(6);
    LOAD(7);
    LOAD(8);
    LOAD(9);
    LOAD(10);
    LOAD(11);
    LOAD(12);
    LOAD(13);
    LOAD(14);
    LOAD(15);

    uniform int elem_num = programCount > 16 ? 16 : programCount;

    // Perform calculation on loaded values
    for (uniform int i = 0; i < elem_num; ++i)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}
