// The test is checking that template name is not clashing with struct member names (#2472)
// RUN: %{ispc} %s --target=host --nostdlib -o %t.o --nowrap 2>&1 | FileCheck %s

// CHECK-NOT: Error: syntax error, unexpected TOKEN_TYPE_NAME
struct MyStruct {
    int M;
    int V;
    int Res1;
    int Res2;
};

template <typename V>
noinline int Sum1(V* a, V* b) {
    return a->M + b->V;
}

template <typename V>
noinline int Sum2(V& a, V& b) {
    return a.M + b.V;
}

template <typename V>
noinline void Res1(V* a, V* b) {
    a->Res1 = a->M + b->V;
    a->Res2 = a->M - b->V;
}

template <typename V>
noinline int Res2(V& a, V& b) {
    b.Res1 = a.M + b.V;
    b.Res2 = a.M - b.V;
}

int foo() {
    MyStruct a = {1, 2, 0};
    MyStruct b = {3, 4, 0};
    Res1(&a, &b);
    Res2(a, b);
    return Sum1(&a, &b) + Sum2(a, b);
}
