// RUN: %{ispc} --target=host --nowrap --nostdlib --emit-llvm-text %s -o - 2>&1 | FileCheck %s

// CHECK-LABEL: @bar___uni
// CHECK-DAG: ret i32 9

// CHECK-NOT: Error: Ambiguous use of overloaded function "fname".

template <typename T> uniform int fname(T x);

uniform int bar(uniform int x) {
    uniform int fname = 9;
    if (fname == 0) {
        return -fname;
    }
    return fname;
}
