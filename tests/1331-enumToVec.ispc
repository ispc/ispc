#include "../test_static.isph"
enum Enum111{
    One,
    Two,
    Three,
    Four
};

// For this to work, an enum->vector type-casting need to happen.
int func(int<5>  x) {
    int ret = 0;
    if(x[0] == Three)
        ret = 7;
    return ret;
}


task void f_v(uniform float RET[]) {
    uniform Enum111 enum_ = Three;
    int rval = func(enum_);
    RET[programIndex] = rval;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 7;
}
