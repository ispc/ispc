// The test checks that for nested template instantiations without arguments works fine.

// RUN: %{ispc} %s --ast-dump --target=host --nostdlib --nowrap | FileCheck %s -check-prefix=CHECK-AST
// RUN: %{ispc} %s --target=host --nostdlib -o %t.o

// CHECK-AST: FunctionTemplate {{.*}} [ varying float()] "DoStuff"
// CHECK-AST: (instantiation <true>) {{.*}}  [ varying float()] "DoStuff"
// CHECK-AST: FunctionTemplate {{.*}} [ varying float()] "KernelInternal"
// CHECK-AST: (instantiation <true>) {{.*}}  [ varying float()] "KernelInternal"

// CHECK-NOT: FATAL ERROR
template <bool b1> inline float DoStuff() { return b1 ? 1 : 2; }

template <bool b1> inline float KernelInternal() {
    float<3> rgb = {0, 0, 0};
    if (b1) {
        return 2 * DoStuff<b1>() + 1;
    }

    return -1;
}

export void Kernel() { KernelInternal<true>(); }
