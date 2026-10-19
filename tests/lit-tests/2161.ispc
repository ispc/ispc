// Regression test for https://github.com/ispc/ispc/issues/2161
// A member pointer to a forward-declared struct can be tested, assigned from
// new, indexed and accessed once the struct is defined, at file and function
// scope.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.o

struct A;

struct B {
    A *a;
};

struct A {
    int r;
};

void fileScope() {
    B b;
    if (b.a)
        b.a->r = 0;
    uniform B ub;
    ub.a = uniform new uniform A[10];
    ub.a[2].r = 1;
}

void functionScope() {
    struct LA;
    struct LB {
        LA *a;
    };
    struct LA {
        int r;
    };
    uniform LB b;
    b.a = uniform new uniform LA[10];
    b.a[2].r = 1;
}
