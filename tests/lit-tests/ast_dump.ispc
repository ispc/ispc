// RUN: %{ispc} %s --target=host -O0 --ast-dump | FileCheck %s -check-prefix=CHECK_USER
// RUN: %{ispc} %s --target=host -O0 --ast-dump=user | FileCheck %s -check-prefix=CHECK_USER
// RUN: %{ispc} %s --target=host -O0 --ast-dump=all | FileCheck %s -check-prefix=CHECK_USER

// "all" is stdlib function.
// After target redesign, stdlib is not parsed anymore alongside with user supplied code.

// CHECK_USER-NOT: Function {{.*}} "all"
// CHECK_USER: Function {{.*}} "foo"

int foo(int a, int b) {
  return a+b;
}
