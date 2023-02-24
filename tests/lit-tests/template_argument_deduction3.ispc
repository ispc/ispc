// The test check template argument deduction rules with different combination of const and *.
// The test is designed to be a valid C++ program, and can be compiled by C++ compiler to observe C++ template deduction rules.
// RUN: %{ispc} %s --nostdlib --target=host --emit-llvm-text --wno-perf -O0 -o - | FileCheck %s

#ifdef ISPC
#define NOINLINE noinline
#define UNMASKED unmasked
#define VARYING varying
#else
#define NOINLINE
#define UNMASKED
#define VARYING
#endif

template <typename T> NOINLINE VARYING T foo1(T * t) { return *t; }
template <typename T> NOINLINE VARYING T foo2(const T * t) { return *t; }
template <typename T> NOINLINE VARYING T foo3(T * const t) { return *t; }
template <typename T> NOINLINE VARYING T foo4(const T * const t) { return *t; }

UNMASKED void bar(int *p, const int *cp, int * const pc, const int * const cpc) {
    // CHECK: call {{.*}} @foo1___uni___vy_3C_uni_3E_
    // CHECK: call {{.*}} @foo1___Cuni___vy_3C_Cuni_3E_
    // CHECK: call {{.*}} @foo1___uni___vy_3C_uni_3E_
    // CHECK: call {{.*}} @foo1___Cuni___vy_3C_Cuni_3E_
    foo1(p);    // T = int
    foo1(cp);   // T = const int
    foo1(pc);   // T = int
    foo1(cpc);  // T = const int

    // CHECK-COUNT-4: call {{.*}} @foo2___uni___vy_3C_Cuni_3E_
    foo2(p);    // T = int
    foo2(cp);   // T = int
    foo2(pc);   // T = int
    foo2(cpc);  // T = int

    // CHECK: call {{.*}} @foo3___uni___vy_3C_uni_3E_
    // CHECK: call {{.*}} @foo3___Cuni___vy_3C_Cuni_3E_
    // CHECK: call {{.*}} @foo3___uni___vy_3C_uni_3E_
    // CHECK: call {{.*}} @foo3___Cuni___vy_3C_Cuni_3E_
    foo3(p);    // T = int
    foo3(cp);   // T = const int
    foo3(pc);   // T = int
    foo3(cpc);  // T = const int

    // CHECK-COUNT-4: call {{.*}} @foo4___uni___vy_3C_Cuni_3E_
    foo4(p);    // T = int
    foo4(cp);   // T = int
    foo4(pc);   // T = int
    foo4(cpc);  // T = int
}

