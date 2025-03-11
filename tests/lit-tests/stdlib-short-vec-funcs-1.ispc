// This test just checks that abs and similar functions with short vectors are
// compiled without errors for all types and reasonalbe vector sizes.

// RUN: %{ispc} --target=host -o %t.o %s 2>&1

#define VARYING(NAME, TYPE, N)                                                                                         \
    varying TYPE<N> varying_##NAME(varying TYPE<N> a) { return NAME<TYPE, N>(a); }

#define UNIFORM(NAME, TYPE, N)                                                                                         \
    uniform TYPE<N> uniform_##NAME(uniform TYPE<N> a) { return NAME<TYPE, N>(a); }

#define COMMON(NAME, TYPE, N)                                                                                          \
    TYPE<N> common_##NAME(TYPE<N> a) { return NAME<TYPE, N>(a); }

#define VARYING_DED(NAME, TYPE, N)                                                                                     \
    varying TYPE<N> varying_##NAME##_ded(varying TYPE<N> a) { return NAME(a); }

#define UNIFORM_DED(NAME, TYPE, N)                                                                                     \
    uniform TYPE<N> uniform_##NAME##_ded(uniform TYPE<N> a) { return NAME(a); }

#define COMMON_DED(NAME, TYPE, N)                                                                                      \
    TYPE<N> common_##NAME##_ded(TYPE<N> a) { return NAME(a); }

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
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, int16)                                                                           \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, int32)                                                                           \
    FUNCS_WITH_DIFFERENT_WIDTHS(NAME, int64)

//  abs is not defined for unsigned types in standard library
//     FUNCS_WITH_DIFFERENT_WIDTHS(NAME, uint8)                                                                           \
//     FUNCS_WITH_DIFFERENT_WIDTHS(NAME, uint16)                                                                          \
//     FUNCS_WITH_DIFFERENT_WIDTHS(NAME, uint32)                                                                          \
//     FUNCS_WITH_DIFFERENT_WIDTHS(NAME, uint64)                                                                          \

FUNCS_NAME(abs)
