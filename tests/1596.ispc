static uniform int sz = sizeof(uniform int);

static uniform int a = 7;
static uniform int * uniform pa = &a;
static const uniform int * uniform parr[] = {NULL, &a, &a};

static const uniform int b[] = {2, 9, 3, 4, 8, 5, 6, 3};
static const uniform int * uniform pb[] = {NULL, b + 5, b};
static const uniform int * uniform pb1[] = {NULL, b, b};

export uniform int width() { return programCount; }

export void f_v(uniform float RET[]) {

    RET[programIndex] = *pa * *parr[2] * *pb[1] * *pb1[2];
    //print("\n Calc Val = % Orig Val = % \n", *pa * *parr[2] * *pb[1] * *pb1[2], 7 * 7 * 5 * 2);

}

export void result(uniform float RET[]) { RET[programIndex] = 490; }
