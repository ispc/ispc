#include "../test_static.isph"
// rule: skip on arch=xe64
struct Point { float x, y, z; };


static void p(uniform float *uniform ptr) {
//CO    for (uniform int s = 0; s < 1; ++s) { // num to print
//CO        for (uniform int i = 0; i < 3; ++i) { // num float in unif struct
//CO            for (uniform int j = 0; j < 8; ++j, ++ptr)  // soa width
//CO                print("% ", *ptr);
//CO            print("\n");
//CO        }
//CO        print("\n");
//CO    }
}

soa<8> Point * uniform aossoa(uniform Point aospts[], uniform int count) {
    uniform int roundUp = (count + 7) & ~0x7;
    uniform int nAlloc = roundUp / 8;

    soa<8> Point * uniform ret = uniform new soa<8> Point[nAlloc];
    foreach (i = 0 ... count) {
//CO        varying Point gp = { programIndex+1, 2*programIndex+1, 3*programIndex+1 };
//CO        ret[i] = gp;
//CO        ret[i].x = gp.x;
//CO        ret[i].y = gp.y;
//CO        ret[i].z = gp.z;
//CO        print("%: % % %\n", i, gp.x, gp.y, gp.z);
            #pragma ignore warning(perf)
            ret[i] = aospts[i];
    }

//CO    p((uniform float * uniform)aospts);
//CO    print("----\n");
//CO    p((uniform float * uniform)ret);

    return ret;

}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    uniform Point pts[programCount+4]; 
    foreach (i = 0 ... programCount+4) {
        #pragma ignore warning(perf)
        pts[i].x = b*i;
        #pragma ignore warning(perf)
        pts[i].y = 2*b*i;
        #pragma ignore warning(perf)
        pts[i].z = 3*b*i;
    }

    soa<8> Point * uniform soaPts = aossoa(pts, programCount+4);

    #pragma ignore warning(perf)
    RET[programIndex] = soaPts[programIndex+3].z;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 15*(programIndex+3);
}
