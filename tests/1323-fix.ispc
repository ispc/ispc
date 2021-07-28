#include "../test_static.isph"
// This test is written to cover the scenario from issue 1323.
// Structs with uniform vector elements directly or as nested elements fail
// compilation when compiled for multiple targets. To test this use case, this
// file needs to be compiled manually for all multiple targets
typedef float<3> float3;

// struct with uniform vector element
struct AABB {
    uniform float3 pmin;
};

// struct with array of vector element
struct AABB1 {
    uniform float3 pmin1[3];
};

// struct with nested struct element with vector element
struct AABB2 {
    AABB ab;
};

// Returns first element from vector
static float check_x(uniform float3 a) {
    return a[0];
}

// Returns first vector element of first array element
static float check_x1(uniform float3 a[]) {
    return a[0][0];
}


task void f_v(uniform float RET[]) {

    AABB box;
    // Sets value of first vector element value
    box.pmin[0]=2.0;
    AABB1 box1;
    // Sets value of first vector element of first array element
    box1.pmin1[0][0] = 5.0;
    AABB2 box2;
    // Sets first element of nested struct
    box2.ab.pmin[0] = 9.0;

    // The following should set all elements of RET to 16.0
    RET[programIndex] = check_x((box.pmin));
    RET[programIndex] += check_x1((box1.pmin1));
    RET[programIndex]+=check_x((box2.ab.pmin));

}

task void result(uniform float RET[]) {
    RET[programIndex] = 16.0;
}
