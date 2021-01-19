// RUN: %{ispc} --target=host --nowrap %s -o %t.o

typedef struct {
    float x;
    int y;
} float_int8_t;

struct main_struct{
    struct {
        uniform float xs;
        uniform int ys;
    } a;
} main_struct_t;

float_int8_t init(float x, int y) {
    y += main_struct_t.a.ys;
    float_int8_t out = {x, y};
    return out;
}
