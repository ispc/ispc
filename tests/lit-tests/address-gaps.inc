// Shared source of the address-gap*.ispc tests: kernels whose gathers,
// scatters, vector loads and stores have byte offsets of 2 GB or more, or
// offsets that wrap at 2^32, and a C driver that runs the cases named on its
// command line. The driver reserves 8 GB without access and makes small
// windows readable and writable at p - 2 GB, around p, around p + 2 GB and just
// below p + 4 GB, so that an address that is 4 GB off faults or reads another
// window. Each case prints "<name> ok" or "<name> wrong"; the driver fails if
// a case is wrong or faults.

#ifdef ISPC
// The driver checks that the kernels see the same gang width.
export uniform int width() { return programCount; }

// Lane 0 is off. Its signed byte offset, 4 * b, doesn't fit in 32 bits; the
// offsets of the lanes that are on do. The function mask is all on, so the
// mask is the constant programIndex > 0.
export void signed_load(uniform float r[], uniform float p[], uniform int b) {
    float v = -1.0f;
    if (programIndex > 0)
        v = p[b + programIndex];
    r[programIndex] = v;
}

export void signed_store(uniform float p[], uniform int b) {
    if (programIndex > 0)
        p[b + programIndex] = 6.0f;
}

// Lane 0 is off. Its index, b - 1, would overflow; the index is only computed
// for the lanes that are on.
export void signed_load_i8(uniform int8 r[], uniform int8 p[], uniform int b) {
    int8 v = -1;
    if (programIndex > 0)
        v = p[b + programIndex - 1];
    r[programIndex] = v;
}

// All lanes are on. The unsigned index b + programIndex wraps from 2^32 - 1
// to 0 between two lanes, so the lanes don't access contiguous memory.
export void wrap_load_i8(uniform int8 r[], uniform int8 p[], uniform unsigned int b) {
    r[programIndex] = p[b + programIndex];
}

export void wrap_store_i8(uniform int8 p[], uniform unsigned int b) { p[b + programIndex] = (int8)(40 + programIndex); }

export void foreach_wrap_load_i8(uniform int8 r[], uniform int8 p[], uniform unsigned int b) {
    foreach (k = 0 ... programCount)
        r[k] = p[b + k];
}

// The signed form: b + programIndex wraps from INT32_MAX to INT32_MIN. Only
// defined with --wrap-signed-int.
export void signed_wrap_load_i8(uniform int8 r[], uniform int8 p[], uniform int b) {
    r[programIndex] = p[b + programIndex];
}

// p[idx - 1] with a signed idx: (idx - 1) * 4 fits in 32 bits, idx * 4 doesn't.
export void minus1_gather(uniform float r[], uniform float p[], uniform int idx[]) {
    r[programIndex] = p[idx[programIndex] - 1];
}

export void minus1_scatter(uniform float p[], uniform int idx[]) { p[idx[programIndex] - 1] = 7.0f + programIndex; }

// A struct member at byte offset 2^31 (an array may have at most 2^31 - 1
// elements, so the padding takes two).
struct Big {
    int8 pad0[0x40000000];
    int8 pad1[0x40000000];
    float x;
};

export void struct_member(uniform float r[], uniform Big *uniform s) { r[programIndex] = s->x; }

export void struct_member_store(uniform Big *uniform s) { s->x = 9.0f; }

// The same member through a varying pointer: a gather with a constant offset.
export void struct_member_varying(uniform float r[], uniform Big *uniform s, uniform int z[]) {
    Big *varying q = s + z[programIndex];
    r[programIndex] = q->x;
}
#else
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#define GB (1ull << 30)
#define WINDOW (1 << 16)

static char *P;
static int W;

static int64_t wrap_index(uint32_t b, int j) { return (int64_t)(uint32_t)(b + (uint32_t)j); }
static int64_t signed_wrap_index(int b, int j) { return (int64_t)(int32_t)((uint32_t)b + (uint32_t)j); }

static int c_signed_load(void) {
    float *p = (float *)P, r[64];
    int b = -(1 << 29) - 1, ok;
    for (int j = 1; j < W; j++)
        p[b + j] = 60.0f + j;
    signed_load(r, p, b);
    ok = r[0] == -1.0f;
    for (int j = 1; j < W; j++)
        ok &= r[j] == 60.0f + j;
    return ok;
}

static int c_signed_store(void) {
    float *p = (float *)P;
    int b = -(1 << 29) - 1, ok = 1;
    signed_store(p, b);
    for (int j = 1; j < W; j++)
        ok &= p[b + j] == 6.0f;
    return ok;
}

static int c_signed_load_i8(void) {
    int8_t *p = (int8_t *)P, r[64];
    int ok;
    for (int j = 1; j < W; j++)
        p[(int64_t)INT_MIN + j - 1] = (int8_t)(70 + j);
    signed_load_i8(r, p, INT_MIN);
    ok = r[0] == -1;
    for (int j = 1; j < W; j++)
        ok &= r[j] == 70 + j;
    return ok;
}

static int c_wrap_load_i8(void) {
    int8_t *p = (int8_t *)P, r[64];
    uint32_t b = 0xFFFFFFFEu;
    int ok = 1;
    for (int j = 0; j < W; j++)
        p[wrap_index(b, j)] = (int8_t)(20 + j);
    wrap_load_i8(r, p, b);
    for (int j = 0; j < W; j++)
        ok &= r[j] == 20 + j;
    return ok;
}

static int c_wrap_store_i8(void) {
    int8_t *p = (int8_t *)P;
    uint32_t b = 0xFFFFFFFEu;
    int ok = 1;
    wrap_store_i8(p, b);
    for (int j = 0; j < W; j++)
        ok &= p[wrap_index(b, j)] == 40 + j;
    return ok;
}

static int c_foreach_wrap_load_i8(void) {
    int8_t *p = (int8_t *)P, r[64];
    uint32_t b = 0xFFFFFFFEu;
    int ok = 1;
    for (int j = 0; j < W; j++)
        p[wrap_index(b, j)] = (int8_t)(30 + j);
    foreach_wrap_load_i8(r, p, b);
    for (int j = 0; j < W; j++)
        ok &= r[j] == 30 + j;
    return ok;
}

static int c_signed_wrap_load_i8(void) {
    int8_t *p = (int8_t *)P, r[64];
    int b = INT_MAX - 1, ok = 1;
    for (int j = 0; j < W; j++)
        p[signed_wrap_index(b, j)] = (int8_t)(50 + j);
    signed_wrap_load_i8(r, p, b);
    for (int j = 0; j < W; j++)
        ok &= r[j] == 50 + j;
    return ok;
}

static int c_minus1_gather(void) {
    float *p = (float *)P, r[64];
    int idx[64], ok = 1;
    for (int j = 0; j < W; j++) {
        idx[j] = (1 << 29) - 3 * j;
        p[(int64_t)idx[j] - 1] = 80.0f + j;
    }
    minus1_gather(r, p, idx);
    for (int j = 0; j < W; j++)
        ok &= r[j] == 80.0f + j;
    return ok;
}

static int c_minus1_scatter(void) {
    float *p = (float *)P;
    int idx[64], ok = 1;
    for (int j = 0; j < W; j++)
        idx[j] = (1 << 29) - 3 * j;
    minus1_scatter(p, idx);
    for (int j = 0; j < W; j++)
        ok &= p[(int64_t)idx[j] - 1] == 7.0f + j;
    return ok;
}

static int c_struct_member(void) {
    float r[64];
    int ok = 1;
    *(float *)(P + (1ull << 31)) = 5.0f;
    struct_member(r, (struct Big *)P);
    for (int j = 0; j < W; j++)
        ok &= r[j] == 5.0f;
    return ok;
}

static int c_struct_member_store(void) {
    struct_member_store((struct Big *)P);
    return *(float *)(P + (1ull << 31)) == 9.0f;
}

static int c_struct_member_varying(void) {
    float r[64];
    int z[64], ok = 1;
    memset(z, 0, sizeof(z));
    *(float *)(P + (1ull << 31)) = 11.0f;
    struct_member_varying(r, (struct Big *)P, z);
    for (int j = 0; j < W; j++)
        ok &= r[j] == 11.0f;
    return ok;
}

static const struct {
    const char *name;
    int (*run)(void);
} cases[] = {
    {"signed_load", c_signed_load},
    {"signed_store", c_signed_store},
    {"signed_load_i8", c_signed_load_i8},
    {"wrap_load_i8", c_wrap_load_i8},
    {"wrap_store_i8", c_wrap_store_i8},
    {"foreach_wrap_load_i8", c_foreach_wrap_load_i8},
    {"signed_wrap_load_i8", c_signed_wrap_load_i8},
    {"minus1_gather", c_minus1_gather},
    {"minus1_scatter", c_minus1_scatter},
    {"struct_member", c_struct_member},
    {"struct_member_store", c_struct_member_store},
    {"struct_member_varying", c_struct_member_varying},
};

static int protect(char *a, size_t n) {
    if (mprotect(a, n, PROT_READ | PROT_WRITE)) {
        perror("mprotect");
        return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    char *reserved = (char *)mmap(NULL, 8 * GB, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (reserved == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    P = reserved + 3 * GB;
    if (!protect(P - 2 * GB, WINDOW) || !protect(P - WINDOW, 2 * WINDOW) || !protect(P + 2 * GB - WINDOW, 2 * WINDOW) ||
        !protect(P + 4 * GB - WINDOW, WINDOW))
        return 1;
    W = width();
    if (W > 64) {
        printf("gang width %d is not supported\n", W);
        return 1;
    }
    int failed = 0;
    for (int i = 1; i < argc; i++) {
        size_t c = 0;
        while (c < sizeof(cases) / sizeof(cases[0]) && strcmp(cases[c].name, argv[i]))
            c++;
        if (c == sizeof(cases) / sizeof(cases[0])) {
            printf("unknown case %s\n", argv[i]);
            return 1;
        }
        // Print the name first, so that a fault shows which case it was.
        printf("%s ", argv[i]);
        fflush(stdout);
        int ok = cases[c].run();
        failed |= !ok;
        printf("%s\n", ok ? "ok" : "wrong");
        fflush(stdout);
    }
    return failed;
}
#endif // ISPC
