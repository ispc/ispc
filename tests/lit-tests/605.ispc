// Regression test for https://github.com/ispc/ispc/issues/605
// A typedef of a pointer to a forward-declared struct can be used to access
// the struct once it is defined.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.o

struct S;
typedef uniform S *SPtr;

struct S {
    int data;
};

void foo(SPtr sptr) { sptr->data = 0; }
