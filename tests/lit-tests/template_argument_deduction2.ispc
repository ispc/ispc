// The test check template argument deduction rules with different combination of const and &.
// The test is designed to be a valid C++ program, and can be compiled by C++ compiler to observe C++ template deduction rules.
// RUN: %{ispc} %s --nostdlib --target=host --emit-llvm-text -O0 -o - | FileCheck %s

#ifdef ISPC
#define NOINLINE noinline
#define UNMASKED unmasked
#else
#define NOINLINE
#define UNMASKED
#endif

template <typename T> NOINLINE T foo1(T t) { return t; }
template <typename T> NOINLINE T foo2(T &t) { return t; }
template <typename T> NOINLINE T foo3(const T t) { return t; }
template <typename T> NOINLINE T foo4(const T &t) { return t; }

UNMASKED void bar(int i, int& ri, const int ci, const int &cri) {
  // CHECK-COUNT-4: call {{.*}} @foo1___vyi___vyi
  foo1(i);   // int(int),               T = int. P = T, A = int.               T = int.
  foo1(ri);  // int(int),               T = int. P = T, A = int&.              A ref dropped before deduction (A = int), adjusted A is int. T = int.
  foo1(ci);  // int(int),               T = int. P = T, A = const int.         cv removal from A (P is not ref), adjusted A is int. T = int.
  foo1(cri); // int(int),               T = int. P = T, A = const int&         A ref dropped before deduction (A = const int), cv removal (P is not ref), adjusted A is int. T = int.

  // CHECK-COUNT-2: call {{.*}} @foo2___vyi___REFvyi
  // CHECK-COUNT-2: call {{.*}} @foo2___Cvyi___REFCvyi
  foo2(i);  // int (int &),             T = int.       P = T&. A = int.        P ref removal. Adjust P = T. T = int.
  foo2(ri); // int (int &),             T = int.       P = T&. A = int&.       A ref dropped before deduction (A = int). P ref removal. Adjust P = T, A = int => T = int.
  foo2(ci); // const int (const int &), T = const int. P = T&. A = const int.  P ref removal. Adjust P = T => T = const int.
  foo2(cri);// const int (const int &), T = const int. P = T&. A = const int&. A ref dropped before deduction (A = const int). P ref removal. Adjust P = T, A = const int => T = const int.

  // CHECK-COUNT-4: call {{.*}} @foo3___vyi___Cvyi
  foo3(i);  // int (const int),         T = int. P = const T. A = int.         cv removal from P. P = int. A = int => T = int.
  foo3(ri); // int (const int),         T = int. P = const T, A = int&.        A ref dropped before deduction. cv removal from P. P = int. A = int => T = int.
  foo3(ci); // int (const int),         T = int. P = const T, A = const int.   cv removal from A (P is not ref), cv removal from P. P = int. A = int => T = int.
  foo3(cri);// int (const int),         T = int. P = const T, A = const int&.  A ref dropped before deduction (A = const int). cv removal from A (P is not ref), cv removal from P. P = int. A = int => T = int.

  // CHECK-COUNT-4: call {{.*}} @foo4___vyi___REFCvyi
  foo4(i);  // int (const int &),       T = int. P = const T&. A = int.        P ref removal. P = const int. A = int. According to [temp.deduct.call]p4*, we can drop const from P => T = int.
  foo4(ri); // int (const int &),       T = int. P = const T&. A = int&.       A ref dropped before deduction (A = int). P ref removal. P = const int. A = int. According to [temp.deduct.call]p4, we can drop const from P => T = int.
  foo4(ci); // int (const int &),       T = int. P = const T&. A = const int.  P ref removal. P = const T. A = const int. Const canceled => T = int.
  foo4(cri);// int (const int &),       T = int. P = const T&. A = const int&. A ref dropped before deduction (A = const int). P ref removal. P = const T. A = const int. Const canceled => T = int.
            //
            // * [temp.deduct.call]p4:
            // If the original P is a reference type, the deduced A (i.e., the type referred to by the reference) can be more cv-qualified than the transformed A.
}
