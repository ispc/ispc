// A global is defined once for all the targets of a multi-target compilation,
// with the initializer of the first target. The initializer may differ
// between the targets: a uniform short vector is padded to a number of
// elements that depends on the data type width, so sizeof(uniform bool<2>)
// is 4 on sse4-i32x4 and 2 on avx2-i64x4, and the other target read the
// wrong size. Such initializers get a warning now; the code of a target
// may still use its own value of a constant global. The bytes in memory are
// compared, so structs, which have a type of their own in the module of each
// target, and the padding of short vectors, which may be undefined, don't
// differ; initializers that hold addresses aren't compared.

// RUN: %{ispc} %s -o %t.o --nowrap --target=sse4-i32x4 2>&1 | FileCheck %s --check-prefix=SINGLE --allow-empty
// RUN: %{ispc} %s -o %t.o --nowrap --target=sse4-i32x4,avx2-i64x4 2>&1 | FileCheck %s --check-prefix=MULTI
// With --werror the warning is an error, and the compilation fails before
// the header is written.
// RUN: echo keep > %t-werror.h
// RUN: not %{ispc} %s -o %t-werror.o -h %t-werror.h --nowrap --werror --target=sse4-i32x4,avx2-i64x4 2>&1 | FileCheck %s --check-prefix=WERROR
// RUN: FileCheck %s --check-prefix=KEEP < %t-werror.h

// REQUIRES: X86_ENABLED

// SINGLE-NOT: {{Error|Warning}}
// WERROR: Error: Initializer for global variable "differ_bool2" differs between the targets
// KEEP: {{^keep$}}
// MULTI-NOT: "same_
// MULTI: Warning: Initializer for global variable "differ_bool2" differs between the targets of the multi-target compilation. The variable holds the value of the first target.
// MULTI-NOT: "same_
// MULTI: Warning: Initializer for global variable "differ_int2" differs between the targets of the multi-target compilation. The variable holds the value of the first target.
// MULTI-NOT: "same_
// MULTI: Warning: Initializer for global variable "differ_array" differs between the targets of the multi-target compilation. The variable holds the value of the first target.
// MULTI-NOT: "same_
// MULTI: Warning: Initializer for global variable "differ_nested" differs between the targets of the multi-target compilation. The variable holds the value of the first target.
// MULTI-NOT: "same_
// MULTI: Warning: Initializer for global variable "differ_vector" differs between the targets of the multi-target compilation. The variable holds the value of the first target.
// MULTI-NOT: "same_
// MULTI: Warning: Initializer for global variable "differ_const" differs between the targets of the multi-target compilation. The variable holds the value of the first target, but the code of each target may use its own value of the constant.
// MULTI-NOT: "same_
// MULTI: Warning: Initializer for global variable "differ_typedef_const" differs between the targets of the multi-target compilation. The variable holds the value of the first target, but the code of each target may use its own value of the constant.
// MULTI-NOT: "same_
uniform unsigned int differ_bool2 = sizeof(uniform bool<2>);
uniform unsigned int same_vec16 = sizeof(uniform int<16>);
uniform unsigned int differ_int2 = sizeof(uniform int<2>);
struct V {
    float x;
    float<3> v[2];
};
uniform unsigned int same_struct = sizeof(uniform V);
uniform int same_array[3] = {1, 2, 3};
uniform int *uniform same_pointer = &same_array[2];
struct Z {
    int x;
};
uniform Z same_zero_struct;
uniform Z same_struct_array[2] = {{1}, {2}};
uniform int<3> same_vec3 = {0, 0, 0};
uniform int<3> same_vec3_zero;
uniform int differ_array[2] = {1, sizeof(uniform int<2>)};
struct N {
    Z z;
};
uniform N differ_nested = {{sizeof(uniform int<2>)}};
uniform float<4> differ_vector = {1, 1, 1, sizeof(uniform int<2>)};
uniform float<4> same_splat = {1, 1, 1, 1};
const uniform int differ_const = sizeof(uniform int<2>);
typedef const uniform int CI;
CI differ_typedef_const = sizeof(uniform int<2>);
