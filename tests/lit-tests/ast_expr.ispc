// RUN: %{ispc} %s --ast-dump --wno-perf --target=host

// This test doesn't check anything specific about format of --ast-dump, but
// it covers most of --ast-dump functionality with respect to Expr and verifies
// that the options doesn't crash the compiler.

void unary(int i) {
    i++;
    i--;
    ++i;
    --i;
    !i;
    ~i;
    -i;
}

int binary(int a, int b) {
    int i = (((a + b) - (b * a)) / b) % a;
    a, b;
    int j = ((a << b) >> b) & (a ^ b) | b;
    bool k = (a < b) || (a <= b) && (a == b) || (a >= b) && (a == b) || (a != b);
    return i + j + (int)k;
}

void assign(int a, int b) {
    a = b;
    b *= a;
    b /= a;
    a %= b;
    a += b;
    a -= b;
    a <<= b;
    a >>= b;
    a &= b;
    a ^= b;
    a |= b;
}

int select(int a, int b, int c, int d) { return (a > b) ? a + b : a - b; }

int call_expr_list(int a, int b) { return select(a, b, a + b, a - b); }

int index(int a[], int b) { return a[b]; }

struct S {
    int fieldA;
    int fieldB;
} s;

int struct_member(struct S *p) { return p->fieldA + s.fieldB; }

int vector_member(int<4> v, int<4> *p) { return v.x + p->x; }

void const_expr() {
    bool b1 = true;
    bool b2 = programIndex % 2 == 0 ? true : false;
    int8 i1 = 1;
    uint8 i2 = 1;
    int16 i3 = 1;
    uint16 i4 = 1;
    int32 i5 = 1;
    uint32 i6 = 1;
    int64 i7 = 1;
    uint64 i8 = 1;
    float16 f1 = 1.f16;
    float f2 = 1.;
    double f3 = 1.d;
#if TARGET_WIDTH == 4
    int i = {1, 8, 1, 9};
#elif TARGET_WIDTH == 8
    int i = {1, 8, 1, 9, 1, 1, 1, 1};
#elif TARGET_WIDTH == 16
    int i = {1, 8, 1, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#elif TARGET_WIDTH == 32
    int i = {1, 8, 1, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#elif TARGET_WIDTH == 64
    int i = {1, 8, 1, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#endif
    int folding = programIndex % 3 == 0 ? ((1 + 2) * 3 - 1) : 9 / 2;
    ;
}

void type_cast(int8 b, int i, float f, double d) {
    int ii = (int)b;
    int8 bb = (int8)i;
    int iii = ((int)f) + ((int)d);
    float ff = ((float)d) + ((float)i);
}

int ref(uniform int i, uniform int *uniform p, uniform int &r) {
    // ReferenceExpr
    uniform int &x = (uniform int &)i;
    // RefDerefExpr
    uniform int y = r;
    // PtrDerefExpr
    uniform int z = *p;
    // AddressOfExpr
    uniform int *pp = &y;
}

int size(float *p) { return sizeof(int) + sizeof(p); }

int allocaaa(uniform int i) {
    void *p = alloca(10);
    void *p2 = alloca(i);
}

task void my_task_foo() {}

void others(int vi, uniform int ui) {
    sync;
    launch[5] my_task_foo();
    NULL;
    new uniform int;
    new uniform float[ui];
    new uniform float[vi];
    int *ptr = new int[100];
    delete[] ptr;
}
