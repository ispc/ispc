#include "test_static.isph"
// Arrays of varying bool are indexed by their size in memory (one byte per
// lane), not by the size of the mask register they are loaded into.

struct S {
    varying bool b[3];
};

task void f_v(uniform float RET[]) {
    varying bool rows[8];
    for (uniform int k = 0; k < 8; k++)
        rows[k] = false;
    rows[3] = programIndex % 2 == 0;
    rows[5] = true;

    // Read with a varying index.
    int i = 3 + 2 * (programIndex % 2);
    bool v = rows[i];
    // Write with a varying index.
    int j = 6 + programIndex % 2;
    rows[j] = true;
    // Pointer difference of varying pointers.
    varying bool *p = &rows[i];
    varying bool *q = &rows[1];
    int diff = (int)(p - q);
    // Constant element offset through a varying pointer.
    uniform S s[2];
    s[0].b[0] = s[0].b[1] = s[0].b[2] = false;
    s[1].b[0] = s[1].b[2] = false;
    s[1].b[1] = true;
    S *varying sp = &s[programIndex % 2];
    bool m = sp->b[1];
    // Constant vector element through a varying pointer.
    varying bool<4> bv[2];
    bv[0].x = bv[0].y = bv[0].z = bv[0].w = false;
    bv[1].x = bv[1].y = bv[1].w = false;
    bv[1].z = true;
    varying bool<4> *varying vp = &bv[programIndex % 2];
    bool z = vp->z;

    int r = 0;
    if (v)
        r += 1;
    if (rows[j] && !rows[13 - j])
        r += 10;
    r += 100 * diff;
    if (m)
        r += 1000;
    if (z)
        r += 10000;
    RET[programIndex] = r;
}

task void result(uniform float RET[]) {
    // Even lanes: rows[3] is true, i = 3, diff = 2, s[0].b[1] is false.
    // Odd lanes: rows[5] is true, i = 5, diff = 4, s[1].b[1] and bv[1].z are true.
    RET[programIndex] = (programIndex % 2 == 0) ? 1 + 10 + 200 : 1 + 10 + 400 + 1000 + 10000;
}
