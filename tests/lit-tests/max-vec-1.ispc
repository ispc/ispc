// This test just checks that max functions with short vectors are compiled
// without errors for all types and reasonalbe vector sizes.

// RUN: %{ispc} -O2 --target=host -o %t.o %s 2>&1

#define VARYING(TYPE, N) varying TYPE<N> varying_max(varying TYPE<N> a, varying TYPE<N> b) { \
  return max<TYPE, N>(a, b); \
}

#define UNIFORM(TYPE, N) uniform TYPE<N> uniform_max(uniform TYPE<N> a, uniform TYPE<N> b) { \
  return max<TYPE, N>(a, b); \
}

#define VARYING_DED(TYPE, N) varying TYPE<N> varying_max_ded(varying TYPE<N> a, varying TYPE<N> b) { \
  return max(a, b); \
}

#define UNIFORM_DED(TYPE, N) uniform TYPE<N> uniform_max_ded(uniform TYPE<N> a, uniform TYPE<N> b) { \
  return max(a, b); \
}

#define COMMON(TYPE, N) TYPE<N> common_max(TYPE<N> a, TYPE<N> b) { \
  return max<TYPE, N>(a, b); \
}

#define COMMON_DED(TYPE, N) TYPE<N> common_max_ded(TYPE<N> a, TYPE<N> b) { \
  return max(a, b); \
}

#define FOO(TYPE, N) \
  COMMON(TYPE, N) \
  COMMON_DED(TYPE, N) \
  VARYING(TYPE, N) \
  UNIFORM(TYPE, N) \
  VARYING_DED(TYPE, N) \
  UNIFORM_DED(TYPE, N)

#define FUNCS(TYPE) \
  FOO(TYPE, 1) \
  FOO(TYPE, 2) \
  FOO(TYPE, 3) \
  FOO(TYPE, 4) \
  FOO(TYPE, 8)

FUNCS(float16)
FUNCS(float)
FUNCS(double)
FUNCS(int8)
FUNCS(uint8)
FUNCS(int16)
FUNCS(uint16)
FUNCS(int32)
FUNCS(uint32)
FUNCS(int64)
FUNCS(uint64)
