// The following test case checks for the store coalescing behavior of the
//   GPU LoadStoreVectorizer for Xe targets.
// Load coalescing is either missed by this pass, or performed earlier by the
//   XeGatherCoalescing pass.
// This test checks for sequences of 4-wide store coalescing, as that appears
//   to be the width targeted by the LoadStoreVectorizer.
// Note: LoadStoreVectorizerPass from LLVM 13.0 uses "undef" as a placeholder
//   for resulting vector and "poison" starting from LLVM 14.0 (llvm/llvm-project@cf284f6)
//   It's safe transformation since poison values will be always overwriten with a different value.

// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=325:325 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_325_LoadStoreVectorizerPass.ll %s --check-prefixes CHECK_ALL,CHECK
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=325:325 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_325_LoadStoreVectorizerPass.ll %s --check-prefixes CHECK_ALL,CHECK

// REQUIRES: XE_ENABLED && LLVM_14_0+

#define LOAD(n)   \
    a[n] = _in[n]

#define NUM 64


// CHECK_ALL-LABEL: @scatter_coalescing_loopunroll4pragma

// CHECK:           %{{.*}} = insertelement <4 x float> {{undef|poison}}, float %_in{{.*}}, i{{(32|64)}} 0
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 1
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 2
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 3
// CHECK:           store <4 x float> %{{.*}}, {{(<4 x float>\*)|ptr}} %{{.*}}

task void scatter_coalescing_loopunroll4pragma(uniform float _out[], uniform float _in[]) {
    uniform float a[NUM];

    // Initialization
    for (uniform int i = 0; i < NUM; i++)
        a[i] = 0.0f;

#pragma unroll 4
    for (uniform int i = 0; i < NUM; i++)
        LOAD(i);

    // Perform calculation on loaded values
    for (uniform int i = 0; i < NUM; i++)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}


// CHECK_ALL-LABEL: @scatter_coalescing_loopunroll4inline

// CHECK:           %{{.*}} = insertelement <4 x float> {{undef|poison}}, float %_in{{.*}}, i{{(32|64)}} 0
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 1
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 2
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 3
// CHECK:           store <4 x float> %{{.*}}, {{(<4 x float>\*)|ptr}} %{{.*}}

task void scatter_coalescing_loopunroll4inline(uniform float _out[], uniform float _in[]) {
    uniform float a[NUM];

    // Initialization
    for (uniform int i = 0; i < NUM; ++i)
        a[i] = 0.0f;

    for (uniform int i = 0; i < NUM; i += 4) {
        if (i + 3 < NUM) {
            LOAD(i);
            LOAD(i + 1);
            LOAD(i + 2);
            LOAD(i + 3);
        }
    }

    // Perform calculation on loaded values
    for (uniform int i = 0; i < NUM; i++)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}


// CHECK_ALL-LABEL: @scatter_coalescing_loopunroll8inline

// CHECK:           %{{.*}} = insertelement <4 x float> {{undef|poison}}, float %_in{{.*}}, i{{(32|64)}} 0
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 1
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 2
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 3
// CHECK:           store <4 x float> %{{.*}}, {{(<4 x float>\*)|ptr}} %{{.*}}
// CHECK:           %{{.*}} = insertelement <4 x float> {{undef|poison}}, float %_in{{.*}}, i{{(32|64)}} 0
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 1
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 2
// CHECK:           %{{.*}} = insertelement <4 x float> %{{.*}}, float %_in{{.*}}, i{{(32|64)}} 3
// CHECK:           store <4 x float> %{{.*}}, {{(<4 x float>\*)|ptr}} %{{.*}}

task void scatter_coalescing_loopunroll8inline(uniform float _out[], uniform float _in[]) {
    uniform float a[NUM];

    // Initialization
    for (uniform int i = 0; i < NUM; i++)
        a[i] = 0.0f;

    for (uniform int i = 0; i < NUM; i += 8) {
        if (i + 7 < NUM) {
            LOAD(i);
            LOAD(i + 1);
            LOAD(i + 2);
            LOAD(i + 3);
            LOAD(i + 4);
            LOAD(i + 5);
            LOAD(i + 6);
            LOAD(i + 7);
        }
    }

    // Perform calculation on loaded values
    for (uniform int i = 0; i < NUM; i++)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}
