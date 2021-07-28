#include "../test_static.isph"
//test for 17631 bug in LLVM.

 
struct s_temp
{
    float temp[64];
};
 
int CompressBlockBC7(int A, uniform float b)
{
    // This declaration caused problem because LLVM inserted
    // _chkstk after declaration and vzeroupper before it's call.
    // A will be in ymm at avx, so we lose a half of it.
    s_temp _state;
    // These two loops are here to prevent elimination of declaration
    for (int i=0; i<64; i++) {
        float ii = i;
        _state.temp[i] = b + sin(ii);
    }
    float r = 0;
    for (int j=0; j<64; j+=9) {
        r += _state.temp[j] + j;
    }

    // Here upper bits of A in ymm can be zeros. This will crash the test.
    int B;
    if (A!=0) {
        B = 20;
    }
    else {
        B = 30;
    }
    if(A == 1) {
        B = r;
    }
    return B;
}
 
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int A = programIndex;
    RET[programIndex] = CompressBlockBC7(A, b);
}
 
task void result(uniform float RET[]) {
    RET[programIndex] = 20;
    RET[0] = 30;
    RET[1] = 292;
}
