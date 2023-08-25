// Check different function specifiers with templates.

// RUN: %{ispc} -DNONE %s --nostdlib --target=host --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_NONE
// RUN: %{ispc} -DEXTERN %s --nostdlib --target=host --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_EXTERN
// RUN: %{ispc} -DSTATIC %s --nostdlib --target=host --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_STATIC
// RUN: %{ispc} -DUNMASKED %s --nostdlib --target=host --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_UNMASKED
// RUN: %{ispc} -DINLINE %s --nostdlib --target=host --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_INLINE
// RUN: not %{ispc} -DTYPEDEF %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_TYPEDEF
// RUN: not %{ispc} -DEXTERN_C %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_EXTERN_C
// RUN: not %{ispc} -DEXTERN_SYCL %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_EXTERN_SYCL
// RUN: not %{ispc} -DEXPORT %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_EXPORT
// RUN: not %{ispc} -DTASK %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_TASK
// RUN: not %{ispc} -DVECTORCALL %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_VECTORCALL
// RUN: not %{ispc} -DREGCALL %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_REGCALL

// CHECK_NONE: define linkonce_odr <{{[0-9]*}} x i32> @goo___vyf___vyfvyf(<{{[0-9]*}} x float> %argGooOne, <{{[0-9]*}} x float> %argGooTwo, <{{[0-9]*}} x {{.*}}> %__mask)
#ifdef NONE
template <typename T> noinline int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

// CHECK_EXTERN: define linkonce_odr <{{[0-9]*}} x i32> @goo___vyf___vyfvyf(<{{[0-9]*}} x float> %argGooOne, <{{[0-9]*}} x float> %argGooTwo, <{{[0-9]*}} x {{.*}}> %__mask)
#ifdef EXTERN
template <typename T> noinline extern int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

// CHECK_STATIC: define internal <{{[0-9]*}} x i32> @goo___vyf___vyfvyf(<{{[0-9]*}} x float> %argGooOne, <{{[0-9]*}} x float> %argGooTwo, <{{[0-9]*}} x {{.*}}> %__mask)
#ifdef STATIC
template <typename T> noinline static int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

// CHECK_UNMASKED: define linkonce_odr <{{[0-9]*}} x i32> @goo___vyf___UM_vyfvyf(<{{[0-9]*}} x float> %argGooOne, <{{[0-9]*}} x float> %argGooTwo)
#ifdef UNMASKED
template <typename T> unmasked noinline int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

// CHECK_INLINE-NOT: @goo___vyf___UM_vyfvyf
#ifdef INLINE
template <typename T> inline int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

// CHECK_TYPEDEF: Illegal "typedef" provided with function template.
#ifdef TYPEDEF
template <typename T> noinline typedef int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

// CHECK_EXTERN_C: Error: Illegal linkage provided with function template.
#ifdef EXTERN_C
template <typename T> noinline extern "C" int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

// CHECK_EXTERN_SYCL: Error: Illegal linkage provided with function template.
#ifdef EXTERN_SYCL
template <typename T> noinline extern "SYCL" int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

// CHECK_EXPORT: Error: 'export' not supported for templates.
#ifdef EXPORT
template <typename T> noinline export int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

// CHECK_TASK: Error: 'task' not supported for templates.
#ifdef TASK
template <typename T> noinline task int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

// CHECK_VECTORCALL: Illegal to use "__vectorcall" qualifier on non-extern function
#ifdef VECTORCALL
template <typename T> noinline __vectorcall int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

// CHECK_REGCALL: Illegal to use "__regcall" qualifier on non-extern function
#ifdef REGCALL
template <typename T> noinline __regcall int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}
#endif

float foo(int argFoo0, float argFoo1) {
    return goo<float>(argFoo0, argFoo1);
}
