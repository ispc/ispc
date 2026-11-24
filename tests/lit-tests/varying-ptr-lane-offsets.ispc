// Known bug (#4051): a varying pointer to varying data points to the start
// of the varying value, and each program instance accesses its own lane of
// it. The frontend doesn't apply the lane offset exactly once:
// - &rows[i] with a varying i and &sp->x through a varying pointer return
//   pointers that already include the lane offset, and dereferencing them
//   adds it again, so every lane but lane 0 reads the wrong lane;
// - storing through such a pointer, as *p = 7, adds no lane offset, so
//   every lane writes lane 0.
// This holds for every element type.
//
// XFAIL until the lane offset is applied exactly once.

// RUN: %{ispc} --pic --target=sse4-i32x4 -O2 --woff -h %t.h %s -o %t.o
// RUN: %{cc} -O2 -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.bin
// RUN: %t.bin

// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST
// XFAIL: *

#ifdef ISPC
struct S {
    int x;
    int y;
};

export uniform int width() { return programCount; }

// Returns 1 in a lane if reading through &rows[i] gives rows[i].
export void read_index(uniform int out[]) {
    varying int rows[4];
    for (uniform int k = 0; k < 4; k++)
        rows[k] = programIndex * 10 + k;
    int i = programIndex % 2;
    varying int *p = &rows[i];
    out[programIndex] = *p == rows[i];
}

// Returns 1 in a lane if reading through &sp->x gives sp->x.
export void read_member(uniform int out[]) {
    varying S s[2];
    s[0].x = 100 + programIndex;
    s[1].x = 200 + programIndex;
    varying S *sp = &s[programIndex % 2];
    varying int *px = &sp->x;
    out[programIndex] = *px == sp->x;
}

// Returns 1 in a lane if storing through p writes the lane's own rows[1].
export void write_deref(uniform int out[]) {
    varying int rows[4];
    for (uniform int k = 0; k < 4; k++)
        rows[k] = 0;
    varying int *p = &rows[1];
    *p = 7;
    out[programIndex] = rows[1] == 7;
}
#else
#include <stdio.h>

static int check(const char *name, void (*f)(int *)) {
    int out[64], w = width(), bad = 0;
    f(out);
    for (int j = 0; j < w; j++)
        bad += out[j] != 1;
    printf("%s: %d of %d lanes wrong\n", name, bad, w);
    return bad;
}

int main(void) {
    int bad = check("read_index", read_index);
    bad += check("read_member", read_member);
    bad += check("write_deref", write_deref);
    return bad != 0;
}
#endif
