// Check how pointer types are handled by template argument deduction.
// RUN: %{ispc} %s --target=host --nostdlib --ast-dump -o %t.o | FileCheck %s

template <typename T> noinline void foo1(T * uniform p1, T ** uniform p2, T *** uniform p3) { }
template <typename T> noinline void foo2(T * p1, T ** p2, T *** p3) { }

void bar(uniform int * uniform up1, uniform int ** uniform up2, uniform int *** uniform up3) {
  // CHECK-COUNT-2: FunctionSymbolExpr {{.*}} [ void(uniform int32 * uniform p1, uniform int32 * uniform * uniform p2, uniform int32 * uniform * uniform * uniform p3)] function name: foo1
  foo1<uniform int>(up1, up2, up3);
  foo1(up1, up2, up3);

  // foo2 assume uniform->varying convertion for arguments.
  // CHECK-COUNT-2: FunctionSymbolExpr {{.*}} [ void(uniform int32 * varying p1, uniform int32 * uniform * varying p2, uniform int32 * uniform * uniform * varying p3)] function name: foo2
  foo2<uniform int>(up1, up2, up3);
  foo2(up1, up2, up3);
}
