#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    RET[programIndex] = 0;
    uint8 a8 = -6;
    if(a8 < 0)
        RET[programIndex] = 1;

    uint16 a16 = -8;
    if(a16 < 0)
        RET[programIndex] = 1;

    uint a = -2;
    if(a < 0)
        RET[programIndex] = 1;

    uint32 a32 = -4;
    if(a32 < 0)
        RET[programIndex] = 1;


    uint64 a64 = -3;
    if(a64 < 0)
        RET[programIndex] = 1;

}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
