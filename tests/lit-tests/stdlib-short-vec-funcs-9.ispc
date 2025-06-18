// This test just checks that the clamp function with short vectors are
// compiled without errors for all types and reasonable vector sizes.

// RUN: %{ispc} --target=host -o %t.o %s 2>&1

#define VARYING(NAME, TYPE, N)                                                                                         \
    varying TYPE<N> varying_##NAME(varying TYPE<N> a, varying TYPE<N> b, varying TYPE<N> c) { return NAME<TYPE, N>(a, b, c); }

#define UNIFORM(NAME, TYPE, N)                                                                                         \
    uniform TYPE<N> uniform_##NAME(uniform TYPE<N> a, uniform TYPE<N> b, uniform TYPE<N> c) { return NAME<TYPE, N>(a, b, c); }

#define COMMON(NAME, TYPE, N)                                                                                          \
    TYPE<N> common_##NAME(TYPE<N> a, TYPE<N> b, TYPE<N> c) { return NAME<TYPE, N>(a, b, c); }

#define VARYING_DED(NAME, TYPE, N)                                                                                     \
    varying TYPE<N> varying_##NAME##_ded(varying TYPE<N> a, varying TYPE<N> b, varying TYPE<N> c) { return NAME(a, b, c); }

#define UNIFORM_DED(NAME, TYPE, N)                                                                                     \
    uniform TYPE<N> uniform_##NAME##_ded(uniform TYPE<N> a, uniform TYPE<N> b, uniform TYPE<N> c) { return NAME(a, b, c); }

#define COMMON_DED(NAME, TYPE, N)                                                                                      \
    TYPE<N> common_##NAME##_ded(TYPE<N> a, TYPE<N> b, TYPE<N> c) { return NAME(a, b, c); }

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
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, double)                                                                          \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, int8)                                                                            \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, uint8)                                                                           \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, int16)                                                                           \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, uint16)                                                                          \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, int32)                                                                           \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, uint32)                                                                          \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, int64)                                                                           \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, uint64)

FUNCS_NAME(clamp)
