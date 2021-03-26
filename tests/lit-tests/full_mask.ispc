// RUN: %{ispc} %s -DISPC_EXPORT --emit-llvm-text --no-discard-value-names -O0 --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_ISPC_EXPORT %s

// RUN: %{ispc} %s -DISPC_UNMASKED --emit-llvm-text --no-discard-value-names -O0 --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_ISPC_UNMASKED %s

// RUN: %{ispc} %s -DISPC_NOQUALIF --emit-llvm-text --no-discard-value-names -O0 --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_ISPC_NOQUALIF %s

// RUN: %{ispc} %s -DISPC_TASK --emit-llvm-text --no-discard-value-names -O0 --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK_ISPC_TASK %s

struct Parameters {
    float *vout;
    int param;
};

#ifdef ISPC_EXPORT
// CHECK_ISPC_EXPORT: define void @simple_export___
// CHECK_ISPC_EXPORT: %full_mask_memory = alloca
// CHECK_ISPC_EXPORT: define void @simple_export(
// CHECK_ISPC_EXPORT-NOT: %full_mask_memory = alloca
export void simple_export(void *uniform _p) {
    Parameters *uniform p = (Parameters * uniform) _p;
    if (programIndex > p->param)
        p->vout[programIndex] = programCount * programIndex;
}
#endif
#ifdef ISPC_UNMASKED
// CHECK_ISPC_UNMASKED: define void @simple_unmasked
// CHECK_ISPC_UNMASKED-NOT: %full_mask_memory = alloca
unmasked void simple_unmasked(void *uniform _p) {
    Parameters *uniform p = (Parameters * uniform) _p;
    if (programIndex > p->param)
        p->vout[programIndex] = programCount * programIndex;
}
#endif
#ifdef ISPC_NOQUALIF
// CHECK_ISPC_NOQUALIF: define void @simple
// CHECK_ISPC_NOQUALIF: %full_mask_memory = alloca
void simple(void *uniform _p) {
    Parameters *uniform p = (Parameters * uniform) _p;
    if (programIndex > p->param)
        p->vout[programIndex] = programCount * programIndex;
}
#endif
#ifdef ISPC_TASK

// CHECK_ISPC_TASK: define void @simple_task
// CHECK_ISPC_TASK-NOT: %full_mask_memory = alloca
// CHECK_ISPC_TASK: define void @simple_entry_point__
// CHECK_ISPC_TASK: %full_mask_memory = alloca
// CHECK_ISPC_TASK: define void @simple_entry_point(
// CHECK_ISPC_TASK-NOT: %full_mask_memory = alloca
task void simple_task(void *uniform _p) {
    Parameters *uniform p = (Parameters * uniform) _p;
    if (programIndex > p->param)
        p->vout[programIndex] = programCount * programIndex;
}
export void simple_entry_point(void *uniform parameters, uniform int dim0, uniform int dim1, uniform int dim2) {
    launch[dim0, dim1, dim2] simple_task(parameters);
}
#endif
