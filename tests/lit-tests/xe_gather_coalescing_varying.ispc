// The following test case produces vector wide loads and stores which are not
//   coalescable.
// For the inline case, the loads and stores are already vectorized.
// For the loop case, the compiler generates a gather/scatter, though this may
//   be improvable with deeper analysis on the induction varible index.
// For further optimization, the coalescing pass (or a dedicated pass) could
//   look into reordering the vector stores and loads for increased
//   memory level parallelism.


// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=321:321 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK
// RUN: %{ispc} %s --target=gen9-x8 --arch=xe64 -h %t.h --emit-llvm-text --debug-phase=321:321 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_321_ispcMemoryCoalescing.ll %s --check-prefixes CHECK_ALL,CHECK

// REQUIRES:  XE_ENABLED
// XFAIL:     XE_ENABLED

#define LOAD(n)     \
    a[n] = _in[n]

#define NUM 64

// CHECK_ALL-LABEL: @gather_coalescing_varyinginline

// CHECK:           %_in_load_ptr{{.*}} = load <{{(8|16)}} x float>, <{{(8|16)}} x float>* %ptr_cast_for_load{{[0-9]*}}
// CHECK:           store <{{(8|16)}} x float> %_in_load_ptr{{.*}}, <{{(8|16)}} x float>* %ptrcast{{[0-9]*}}
// CHECK:           %_in_load_ptr{{.*}} = load <{{(8|16)}} x float>, <{{(8|16)}} x float>* %ptr_cast_for_load{{[0-9]*}}
// CHECK:           store <{{(8|16)}} x float> %_in_load_ptr{{.*}}, <{{(8|16)}} x float>* %ptrcast{{[0-9]*}}
// CHECK:           %_in_load_ptr{{.*}} = load <{{(8|16)}} x float>, <{{(8|16)}} x float>* %ptr_cast_for_load{{[0-9]*}}
// CHECK:           store <{{(8|16)}} x float> %_in_load_ptr{{.*}}, <{{(8|16)}} x float>* %ptrcast{{[0-9]*}}
// CHECK:           %_in_load_ptr{{.*}} = load <{{(8|16)}} x float>, <{{(8|16)}} x float>* %ptr_cast_for_load{{[0-9]*}}
// CHECK:           store <{{(8|16)}} x float> %_in_load_ptr{{.*}}, <{{(8|16)}} x float>* %ptrcast{{[0-9]*}}

task void gather_coalescing_varyinginline(uniform float _out[], uniform float _in[]) {
    uniform float a[NUM];

    // Initialization
    for (uniform int i = 0; i < NUM; i++)
        a[i] = 0.0f;

    LOAD(programIndex);
    LOAD(programIndex + programCount);
    LOAD(programIndex + programCount*2);
    LOAD(programIndex + programCount*3);

    // Perform calculation on loaded values
    for (uniform int i = 0; i < NUM; i++)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}

// The following test case is not expected to have coalescing opportunities.

// CHECK_ALL-LABEL: @gather_coalescing_varyingloop

// CHECK:           %res{{.*}} = {{(tail)?}} call <{{(8|16)}} x float> @llvm.genx.svm.gather.v{{(8|16)}}f32
// CHECK:           call void @llvm.genx.svm.scatter.v{{(8|16)}}i1

task void gather_coalescing_varyingloop(uniform float _out[], uniform float _in[]) {
    uniform float a[NUM];

    // Initialization
    for (uniform int i = 0; i < NUM; i++)
        a[i] = 0.0f;

    for (varying int i = programIndex; i < NUM; i += programCount)
        LOAD(i);

    // Perform calculation on loaded values
    for (uniform int i = 0; i < NUM; i++)
        a[i] *= (i + 1);

    _out[programIndex] = a[programIndex];
}
