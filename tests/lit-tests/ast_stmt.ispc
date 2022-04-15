// RUN: %{ispc} %s --ast-dump --wno-perf --target=host

// This test doesn't check anything specific about format of --ast-dump, but
// it covers most of --ast-dump functionality with respect to Stmt and verifies
// that the options doesn't crash the compiler.

int foo(int a, int b) {
    // DeclStmt
    int result = a + b, result2, result3 = a;
    // IfStmt
    if (a > 2) {
        a = 1;
    } else {
        return 3;
    }

    do {
        a++;
    } while (a > b);
    // ForStmt
    // BreakStmt
    // ContinueStmt
    for (uniform int ui = 0; ui < 16; ui++) {
        int t = a;
        a = b;
        if (t == 5)
            continue;
        if (t == 6)
            break;
        b = t;
    }
    // ExprStmt
    result++;
    // Return
    return result;
}

float bar(uniform int n1, uniform int n2, uniform int n3, float f[]) {
    float r = 0;
    // ForeachStmt
    foreach (i1 = 0...n1, i2 = 0...16) {
        f[i1] += 1;
        r = i1 + i2;
    }
    // ForeachActiveStmt
    foreach_active(i) { r += 5; }

    int count;
    // ForeachUniqueStmt
    foreach_unique(ua in r) { ++count; }

    // UnmaskedStmt
    unmasked { count--; }

    // SwitchStmt
    // BreakStmt
    // CaseStmt
    // DefaultStmt
    switch (n1) {
    case 1:
    case 2:
        count--;
        break;
    case 3:
    case 4:
        count++;
        break;
    default:
        count++;
    }

    return r;
}

int baz(uniform int i) {
// LabeledStmt
A:
    if (i > 5) {
        i--;
        // GotoStmt
        goto A;
    } else {
        i++;
    }
    // PrintStmt
    print("asd %, %\n", i, i);
    print("123");
    // AssertStmt
    assert(i < 5);
    float *uniform buf = uniform new float[programCount];
    // DeleteStmt
    delete buf;
    return i;
}
