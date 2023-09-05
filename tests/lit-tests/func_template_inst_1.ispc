// Check different function specifiers with template explicit instantiations.

// RUN: not %{ispc} -DTYPEDEF %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_TYPEDEF
// RUN: not %{ispc} -DEXTERN_C %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_EXTERN_C
// RUN: not %{ispc} -DEXTERN_SYCL %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_EXTERN_SYCL
// RUN: not %{ispc} -DEXPORT %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_EXPORT
// RUN: not %{ispc} -DTASK %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_TASK
// RUN: not %{ispc} -DVECTORCALL %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_VECTORCALL
// RUN: not %{ispc} -DREGCALL %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_REGCALL

// Primary template
template <typename T> noinline int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}

// CHECK_EXTERN: define linkonce_odr <{{[0-9]*}} x i32> @goo___vyf___vyfvyf(<{{[0-9]*}} x float> %argGooOne, <{{[0-9]*}} x float> %argGooTwo, <{{[0-9]*}} x {{.*}}> %__mask)
#ifdef EXTERN
template noinline extern int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_STATIC: define internal <{{[0-9]*}} x i32> @goo___vyf___vyfvyf(<{{[0-9]*}} x float> %argGooOne, <{{[0-9]*}} x float> %argGooTwo, <{{[0-9]*}} x {{.*}}> %__mask)
#ifdef STATIC
template noinline static int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_UNMASKED: define linkonce_odr <{{[0-9]*}} x i32> @goo___vyf___UM_vyfvyf(<{{[0-9]*}} x float> %argGooOne, <{{[0-9]*}} x float> %argGooTwo)
#ifdef UNMASKED
template noinline unmasked int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_INLINE-NOT: @goo___vyf___UM_vyfvyf
#ifdef INLINE
template inline int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_TYPEDEF: Illegal "typedef" provided with template instantiation.
#ifdef TYPEDEF
template noinline typedef int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_EXTERN_C: Error: Illegal linkage provided with template instantiation.
#ifdef EXTERN_C
template noinline extern "C" int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_EXTERN_SYCL: Error: Illegal linkage provided with template instantiation.
#ifdef EXTERN_SYCL
template noinline extern "SYCL" int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_EXPORT: Error: 'export' not supported for template instantiation.
#ifdef EXPORT
template noinline export int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_TASK: Error: 'task' not supported for template instantiation.
#ifdef TASK
template noinline task int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_VECTORCALL: Illegal to use "__vectorcall" qualifier on non-extern function
#ifdef VECTORCALL
template noinline __vectorcall int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_REGCALL: Illegal to use "__regcall" qualifier on non-extern function
#ifdef REGCALL
template noinline __regcall int goo<int>(int argGooOne, int argGooTwo);
#endif

float foo(int argFoo0, float argFoo1) {
    return goo<int>(argFoo0, argFoo1);
}
