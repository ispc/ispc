// RUN:  %{ispc} --time-trace --time-trace-granularity=450 --target=host -o %T/1926.o %s
// RUN: cat %T/1926.o.json \
// RUN:   | FileCheck %s

// CHECK: "traceEvents"
// CHECK: "detail"
export uniform int foo(uniform int a[], uniform int count) {
    uniform int ret = 0;
    for (uniform int i = 0; i < count; i ++) {
        ret += a[i];
    }
}
