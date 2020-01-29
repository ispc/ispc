// RUN: %{ispc} %s > %t 2>&1
uniform int sz = sizeof(uniform int);

static uniform float a = 1.0;
static uniform float * uniform pa = &a;
static const uniform float * uniform parr[] = {NULL, &a, &a};

static const uniform float b[] = {0.,4., 3., 2., 1.};
static const uniform float * uniform pb[] = {NULL, b+3, b};
static const uniform float * uniform pb1[] = {NULL, b, b};

varying float i = 80;
varying float p_foo[] = {1, 4, 3, 2, 4, 3, 3, 3};
varying float * uniform au = &p_foo[0];


varying float usevals() {

    varying float * varying av = au + i + *pa * *parr[1] * *pb[2] * *pb1[1];
    return *av  ;
}


