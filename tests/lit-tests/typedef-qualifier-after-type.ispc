// A typedef keeps its uniform/varying qualifier when other declaration
// specifiers follow the type name (e.g. "UI const x"). Without it, the type is
// unbound and gets the default variability of the context instead.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - 2>&1 | FileCheck %s --implicit-check-not=Error

typedef uniform int UI;
typedef varying int VI;

// Uniform typedef on a global, which defaults to varying: "a" must stay
// uniform, otherwise returning it as uniform int is a conversion error.
UI const a = 1;
uniform int global() { return a; }

// Uniform typedef on a parameter, which defaults to varying: "x" must stay
// uniform, otherwise returning it as uniform int is a conversion error.
uniform int param(UI const x) { return x; }

// Varying typedef on a pointer target, which defaults to uniform: "*p" must
// stay varying. Compare against sizeof(varying int) rather than a constant,
// as the width of --target=host differs between machines.
// CHECK-LABEL: @pointee___
// CHECK: ret i32 1
uniform int pointee(VI const *uniform p) { return sizeof(*p) == sizeof(varying int) ? 1 : 0; }
