static uniform int sz = sizeof(uniform int);

static uniform int a = 7;
static uniform int *uniform pa = &a;
static const uniform int *uniform parr[] = {NULL, &a, &a};

static const uniform int b[] = {2, 9, 3, 4, 8, 5, 6, 3};
static const uniform int *uniform pb[] = {NULL, b + 5, b};
static const uniform int *uniform pb1[] = {NULL, b, b};

uniform int sVar = sizeof(varying int64);

uniform int ii[2][3][4] = {{{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}}, {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}}};
uniform int *uniform aaa = &ii[1][2][3];

static uniform int sVar_loc_global = sizeof(varying int8);

export void f_v(uniform float RET[]) {
    static uniform int sVar_loc = sizeof(varying int8);
    RET[programIndex] = (1 * *pa) + (2 * *parr[2]) + (3 * *pb[1]) + (4 * *pb1[2]) + (5 * *aaa) + (6 * sz) + (7 * sVar) + (8 * sVar_loc) + (9 * sVar_loc_global);
}

// Expected Values
// *pa = 7
// *parr[2] = 7
// *pb[1] = 5
// *pb1[2] = 2
// *aaa = 4
// sz = 4
// sVar = 8 * WIDTH
// sVar_loc = 1 * WIDTH
// sVar_loc_global = 1 * WIDTH
export void result(uniform float RET[]) {
    RET[programIndex] = (1 * 7) + (2 * 7) + (3 * 5) + (4 * 2) + (5 * 4) + (6 * 4) + (7 * (8 * programCount)) + (8 * (1 * programCount)) + (9 * (1 * programCount));
}
