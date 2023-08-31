// Check different function specifiers with template explicit instantiations.

// RUN: %{ispc} -DEXTERN %s --nostdlib --target=host --nowrap --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_EXTERN
// RUN: %{ispc} -DSTATIC %s --nostdlib --target=host --nowrap --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_STATIC
// RUN: %{ispc} -DUNMASKED %s --nostdlib --target=host --nowrap --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_UNMASKED
// RUN: %{ispc} -DEXTERN_INHERIT %s --nostdlib --target=host --nowrap --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_EXTERN_INHERIT
// RUN: %{ispc} -DSTATIC_INHERIT %s --nostdlib --target=host --nowrap --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_STATIC_INHERIT
// RUN: %{ispc} -DUNMASKED_INHERIT %s --nostdlib --target=host --nowrap --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_UNMASKED_INHERIT
// RUN: %{ispc} -DINLINE %s --nostdlib --target=host --nowrap --emit-llvm-text -O0 --nowrap -o - | FileCheck %s --check-prefixes=CHECK_INLINE
// RUN: not %{ispc} -DEXTERN_ERROR %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_EXTERN_ERROR
// RUN: not %{ispc} -DSTATIC_ERROR %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_STATIC_ERROR
// RUN: not %{ispc} -DTYPEDEF %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_TYPEDEF
// RUN: not %{ispc} -DEXTERN_C %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_EXTERN_C
// RUN: not %{ispc} -DEXTERN_SYCL %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_EXTERN_SYCL
// RUN: not %{ispc} -DEXPORT %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_EXPORT
// RUN: not %{ispc} -DTASK %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_TASK
// RUN: not %{ispc} -DVECTORCALL %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_VECTORCALL
// RUN: not %{ispc} -DREGCALL %s --nostdlib --target=host --nowrap 2>&1 | FileCheck %s --check-prefixes=CHECK_REGCALL

// Primary template
template <typename T> noinline
#if defined(EXTERN) || defined(EXTERN_INHERIT)
    extern
#elif defined(STATIC) || defined(STATIC_INHERIT)
    static
#elif defined(UNMASKED) || defined(UNMASKED_INHERIT)
    unmasked
#endif
    int goo(T argGooOne, T argGooTwo) {
    return argGooOne - argGooTwo;
}

// CHECK_EXTERN_INHERIT: define weak_odr <{{[0-9]*}} x i32> @goo___vyi___vyivyi(<{{[0-9]*}} x i32> %argGooOne, <{{[0-9]*}} x i32> %argGooTwo, <{{[0-9]*}} x {{.*}}> %__mask)
// CHECK_STATIC_INHERIT: define internal <{{[0-9]*}} x i32> @goo___vyi___vyivyi(<{{[0-9]*}} x i32> %argGooOne, <{{[0-9]*}} x i32> %argGooTwo, <{{[0-9]*}} x {{.*}}> %__mask)
// CHECK_UNMASKED_INHERIT: define weak_odr <{{[0-9]*}} x i32> @goo___vyi___UM_vyivyi(<{{[0-9]*}} x i32> %argGooOne, <{{[0-9]*}} x i32> %argGooTwo)
#if defined(EXTERN_INHERIT) || defined(STATIC_INHERIT) || defined(UNMASKED_INHERIT)
template noinline int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_EXTERN: define weak_odr <{{[0-9]*}} x i32> @goo___vyi___vyivyi(<{{[0-9]*}} x i32> %argGooOne, <{{[0-9]*}} x i32> %argGooTwo, <{{[0-9]*}} x {{.*}}> %__mask)
#ifdef EXTERN
template noinline extern int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_STATIC: define internal <{{[0-9]*}} x i32> @goo___vyi___vyivyi(<{{[0-9]*}} x i32> %argGooOne, <{{[0-9]*}} x i32> %argGooTwo, <{{[0-9]*}} x {{.*}}> %__mask)
#ifdef STATIC
template noinline static int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_EXTERN_ERROR: Template instantiation has inconsistent storage class. Consider assigning it to the primary template to inherit it's signature.
#ifdef EXTERN_ERROR
template noinline extern int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_STATIC_ERROR: Template instantiation has inconsistent storage class. Consider assigning it to the primary template to inherit it's signature.
#ifdef STATIC_ERROR
template noinline static int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_UNMASKED: define weak_odr <{{[0-9]*}} x i32> @goo___vyi___UM_vyivyi(<{{[0-9]*}} x i32> %argGooOne, <{{[0-9]*}} x i32> %argGooTwo)
#ifdef UNMASKED
template noinline unmasked int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_UNMASKED_ERROR: Template instantiation has inconsistent "unmasked" specifier. Consider moving the specifier inside the function or assigning it to the primary template to inherit it's signature.
#ifdef UNMASKED_ERROR
template noinline unmasked int goo<int>(int argGooOne, int argGooTwo);
#endif

// CHECK_INLINE-NOT: @goo___vyi___vyivyi
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
