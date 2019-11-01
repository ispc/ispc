#define TARGET_WIDTH_FOUR 4
#define TARGET_WIDTH_EIGHT 8
#define TARGET_WIDTH_SIXTEEN 16


    // Arrays for comparison.
#if TARGET_WIDTH == TARGET_WIDTH_FOUR
    float b_arr[TARGET_WIDTH] = {2, 6, 1, 9};
#elif TARGET_WIDTH == TARGET_WIDTH_EIGHT
    float b_arr[TARGET_WIDTH] = {2, 6, 1, 9, 2, 3, 2, -1};
#elif TARGET_WIDTH == TARGET_WIDTH_SIXTEEN
    float b_arr[TARGET_WIDTH] = {2, 6, 1, 9, 2, 3, 2, -1, 2, 6, 1, 9, 2, 3, 2, -1};
#endif

struct Foo {
    float x;
    float y;
};

export void f_f(uniform float RET[], uniform float aFOO[]) {

    RET[programIndex] = 0;

    // Checking that TARGET_WIDTH macro is equal to programCount
    if (TARGET_WIDTH != programCount)
        RET[programIndex] = 1;

    // Checking that TARGET_ELEMENT_WIDTH macro is defined
#ifndef TARGET_ELEMENT_WIDTH
    RET[programIndex] = 1;
#endif

    // Initializing varying variable using constant expression.
    // Syntax : Assume val1 => varying value for programIndex 1
    // varying datatype variable_name = {val1, val2,....}
#if TARGET_WIDTH == TARGET_WIDTH_FOUR
    float b1 = {2, 6, 1, 9};
    if (b_arr[programIndex] != b1)
        RET[programIndex] = 1;
#elif TARGET_WIDTH == TARGET_WIDTH_EIGHT
    float b1 = {2, 6, 1, 9, 2, 3, 2, -1};
    if (b_arr[programIndex] != b1)
        RET[programIndex] = 1;
#elif TARGET_WIDTH == TARGET_WIDTH_SIXTEEN
    float b1 = {2, 6, 1, 9, 2, 3, 2, -1, 2, 6, 1, 9, 2, 3, 2, -1};
    if (b_arr[programIndex] != b1)
        RET[programIndex] = 1;
#endif

    // Initializing array of varying values.
    // Syntax : Assume val1 => varying value for programIndex 1 and we are
    // initializing variable_name[1] with varying initializer
    // varying datatype variable_name[] = {a, {val1, val2,....}}
    #if TARGET_WIDTH == TARGET_WIDTH_FOUR
    float b2[] = {2, {2, 6, 1, 9}, -1};
    if (b2[1] != b_arr[programIndex])
        RET[programIndex] = 1;
#elif TARGET_WIDTH == TARGET_WIDTH_EIGHT
    float b2[] = {2, {2, 6, 1, 9, 2, 3, 2, -1}, -1};
    if (b2[1] != b_arr[programIndex])
        RET[programIndex] = 1;
#elif TARGET_WIDTH == TARGET_WIDTH_SIXTEEN
    float b2[] = {2, {2, 6, 1, 9, 2, 3, 2, -1, 2, 6, 1, 9, 2, 3, 2, -1}, -1};
    if (b2[1] != b_arr[programIndex])
        RET[programIndex] = 1;
#endif

    // Initializing varying as part of struct.
    // Syntax : Assume val1 => varying value for programIndex 1 and we are
    // initializing element y of struct with 2 elements, x and y with varying initializer.
    // varying datatype struct_name variable_name = {a, {val1, val2,....}}
    #if TARGET_WIDTH == TARGET_WIDTH_FOUR
    varying Foo b3 = {2, {2, 6, 1, 9}};
    if (b3.y != b_arr[programIndex])
        RET[programIndex] = 1;
#elif TARGET_WIDTH == TARGET_WIDTH_EIGHT
    varying Foo b3 = {2, {2, 6, 1, 9, 2, 3, 2, -1}};
    if (b3.y != b_arr[programIndex])
        RET[programIndex] = 1;
#elif TARGET_WIDTH == TARGET_WIDTH_SIXTEEN
    varying Foo b3 = {2, {2, 6, 1, 9, 2, 3, 2, -1, 2, 6, 1, 9, 2, 3, 2, -1}};
    if (b3.y != b_arr[programIndex])
        RET[programIndex] = 1;
#endif

    // Initializing varying as part of vector.
    // Syntax : Assume val1 => varying value for programIndex 1 and we are
    // initializing 2nd element of vector with varying initializer
    // datatype<2> variable_name = {a, {val1, val2,....}}
    #if TARGET_WIDTH == TARGET_WIDTH_FOUR
    varying float<2> b4 = {2, {2, 6, 1, 9}};
    if (b4[1] != b_arr[programIndex])
        RET[programIndex] = 1;
#elif TARGET_WIDTH == TARGET_WIDTH_EIGHT
    varying float<2> b4 = {2, {2, 6, 1, 9, 2, 3, 2, -1}};
    if (b4[1] != b_arr[programIndex])
        RET[programIndex] = 1;
#elif TARGET_WIDTH == TARGET_WIDTH_SIXTEEN
    varying float<2> b4 = {2, {2, 6, 1, 9, 2, 3, 2, -1, 2, 6, 1, 9, 2, 3, 2, -1}};
    if (b4[1] != b_arr[programIndex])
        RET[programIndex] = 1;
#endif

}

export void result(uniform float RET[]) { RET[programIndex] = 0; }
