// RUN: %{ispc} %s --target=host -O0 --ast-dump | FileCheck %s -check-prefix=CHECK_USER
// RUN: %{ispc} %s --target=host -O0 --ast-dump=user | FileCheck %s -check-prefix=CHECK_USER
// RUN: %{ispc} %s --target=host -O0 --ast-dump=all | FileCheck %s -check-prefix=CHECK_ALL

// "all" is stdlib function.

// CHECK_USER-NOT: Function {{.*}} "all"
// CHECK_USER: Function {{.*}} "foo"

// CHECK_ALL: Function {{.*}} "all"
// CHECK_ALL: Function {{.*}} "foo"
int foo(int a, int b) {
  return a+b;
}
