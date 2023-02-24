// Test member expressions with dependent types - they need to compile, not crash.
// RUN: %{ispc} %s --target=host --nostdlib -o %t.o

struct S {
  float f1[4];
  float<4> f2;
};

template <typename T> noinline float templ(const T& v1, const T* v2) {
  float res = v1.f1[1]  + v1.f2[1]  + v1.f2.x +
              v2->f1[1] + v2->f2[1] + v2->f2.x;
  return res;
}

float foo (struct S s1, struct S s2) {
  return templ(s1, &s1);
}
