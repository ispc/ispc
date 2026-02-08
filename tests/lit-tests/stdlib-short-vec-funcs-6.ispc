// This test just checks that the target math functions with short vectors are
// compiled without errors for all types and reasonable vector sizes.

// RUN: %{ispc} --target=host -o %t.o %s 2>&1

// LLVM 20 PowerPC backend crashes on vector atan2/pow/fmod with
// "Do not know how to promote this operator's result!". Fixed in LLVM 21.
// UNSUPPORTED: PPC64LE_HOST && !LLVM_21_0+

#include <short_vec.isph>

#define VARYING(NAME, TYPE, N)                                                                                         \
    varying TYPE<N> varying_##NAME(varying TYPE<N> a, varying TYPE<N> b) { return NAME<TYPE, N>(a, b); }

#define UNIFORM(NAME, TYPE, N)                                                                                         \
    uniform TYPE<N> uniform_##NAME(uniform TYPE<N> a, uniform TYPE<N> b) { return NAME<TYPE, N>(a, b); }

#define COMMON(NAME, TYPE, N)                                                                                          \
    TYPE<N> common_##NAME(TYPE<N> a, TYPE<N> b) { return NAME<TYPE, N>(a, b); }

#define VARYING_DED(NAME, TYPE, N)                                                                                     \
    varying TYPE<N> varying_##NAME##_ded(varying TYPE<N> a, varying TYPE<N> b) { return NAME(a, b); }

#define UNIFORM_DED(NAME, TYPE, N)                                                                                     \
    uniform TYPE<N> uniform_##NAME##_ded(uniform TYPE<N> a, uniform TYPE<N> b) { return NAME(a, b); }

#define COMMON_DED(NAME, TYPE, N)                                                                                      \
    TYPE<N> common_##NAME##_ded(TYPE<N> a, TYPE<N> b) { return NAME(a, b); }

#define FUNC_TYPE_WIDTH(NAME, TYPE, N)                                                                                 \
    COMMON(NAME, TYPE, N)                                                                                              \
    COMMON_DED(NAME, TYPE, N)                                                                                          \
    VARYING(NAME, TYPE, N)                                                                                             \
    UNIFORM(NAME, TYPE, N)                                                                                             \
    VARYING_DED(NAME, TYPE, N)                                                                                         \
    UNIFORM_DED(NAME, TYPE, N)

#define FUNCS_WITH_DIFFERENT_WIDTHS(NAME, TYPE)                                                                        \
    FUNC_TYPE_WIDTH(NAME, TYPE, 1)                                                                                     \
    FUNC_TYPE_WIDTH(NAME, TYPE, 2)                                                                                     \
    FUNC_TYPE_WIDTH(NAME, TYPE, 3)                                                                                     \
    FUNC_TYPE_WIDTH(NAME, TYPE, 4)                                                                                     \
    FUNC_TYPE_WIDTH(NAME, TYPE, 8)

#define FUNCS_NAME(NAME)                                                                                               \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, float16)                                                                         \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, float)                                                                           \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, double)

FUNCS_NAME(atan2)
FUNCS_NAME(pow)
FUNCS_NAME(fmod)
