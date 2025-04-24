// This test checks the behavior of the #pragma ignore warning directive

// RUN: %{ispc} %s --nowrap --target=host --emit-llvm-text -o - 2>&1 | FileCheck %s

#define foo(x) ((int8)x / (programIndex-1))

inline int fill(uniform float b[], int8 a) {
    for (int i=0; i < programCount; i++) {
        // Line 11 would have resulted in a performance and a compiler warning.
        // The '#pragma ignore warning(all)' directive ensures that BOTH warnings are ignored.
// CHECK-NOT: Warning: Undefined behavior: all program instances are writing to the same location!
// CHECK-NOT: Warning: Performance Warning: Scatter required to store value.
        #pragma ignore warning(all)
        b[i] = i;
    }
    return b[0];
}

task void f_f(uniform float RET[], uniform float fFOO[]) {
    float f = fFOO[programIndex];
    uniform float a[programCount], b[programCount];
    for (int i=0; i < programCount; i++) {
        // Line 23 would have resulted in a performance and a compiler warning.
        // The '#pragma   ignore   warning ( all )' directive ensures that BOTH warnings are ignored.
// CHECK-NOT: Warning: Undefined behavior: all program instances are writing to the same location!
// CHECK-NOT: Warning: Performance Warning: Scatter required to store value.
        #pragma   ignore   warning ( all )
        a[i] = i;
    }
    int x = 2;
    // Line 29 would have resulted in a performance and a compiler warning.
    // The '#pragma ignore warning(perf)' directive ignores ONLY performance warning.
    if(programIndex + 1 < programCount) {
// CHECK-NOT: Warning: Array index "16" may be out of bounds for 16 element array.
// CHECK-NOT: Performance Warning: Division with varying integer types is very inefficient.
        #pragma ignore warning
        a[programIndex + 1] = fill(b, (int8)x / (programIndex-1));
    }

    // Line 34 would have resulted in a performance warning when the macro 'foo' gets expanded.
    // The '#pragma ignore warning(perf)' directive ignores the performance warning from the expanded macro.
// CHECK-NOT: Performance Warning: Division with varying integer types is very inefficient.
    #pragma ignore warning(perf)
    x = foo(x);

    // Line 39 would have resulted in a compiler warning.
    // The '#pragma ignore warning' directive ignores this warning.
// CHECK-NOT: Warning: Array index "16" may be out of bounds for 16 element array. 
    #pragma ignore warning
    a[programIndex + 1] =  b[programIndex];
}
