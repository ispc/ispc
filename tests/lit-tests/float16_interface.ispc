// RUN: %{ispc} %s --target=host -h %t.h
// RUN: FileCheck --input-file=%t.h %s

// CHECK: extern void export_float16(__fp16 arg0, __fp16 * arg1);

export void export_float16(uniform float16 arg0, uniform float16 arg1[]) {}
