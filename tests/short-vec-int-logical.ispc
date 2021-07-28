#include "../test_static.isph"
typedef int<3> int3;


// a[] = { 1, 2, 3, ... };
task void f_f(uniform float RET[], uniform float a[]) {
    RET[programIndex] = 0;
    int int_var = a[programIndex];

    int3 int_vec1 = {int_var - 1, int_var - 2, int_var - 3};
    int3 int_vec2 = {int_var - 2, int_var - 2, int_var - 3};

    int3 int_vec = int_vec1 || int_vec2;
    // Check result of Logical OR when both operands are short vectors.
    if ((int_vec.x != ((int_var - 1) || (int_var - 2))) || (int_vec.y != ((int_var - 2) || (int_var - 2))) ||
        (int_vec.z != ((int_var - 3) || (int_var - 3))))
        RET[programIndex] = -1;

    int_vec = (int_var - 1) || int_vec1;
    // Check result of Logical OR when 2nd operand is a short vector.
    if ((int_vec.x != ((int_var - 1) || (int_var - 1))) || (int_vec.y != ((int_var - 2) || (int_var - 1))) ||
        (int_vec.z != ((int_var - 3) || (int_var - 1))))
        RET[programIndex] = -2;

    int_vec = int_vec1 || (int_var - 1);
    // Check result of Logical OR when 1st operand is a short vector.
    if ((int_vec.x != ((int_var - 1) || (int_var - 1))) || (int_vec.y != ((int_var - 2) || (int_var - 1))) ||
        (int_vec.z != ((int_var - 3) || (int_var - 1))))
        RET[programIndex] = -3;

    int_vec = int_vec1 && int_vec2;
    // Check result of Logical AND when both operands are short vectors.
    if ((int_vec.x != ((int_var - 1) && (int_var - 2))) || (int_vec.y != ((int_var - 2) && (int_var - 2))) ||
        (int_vec.z != ((int_var - 3) && (int_var - 3))))
        RET[programIndex] = -4;

    int_vec = (int_var - 1) && int_vec1;
    // Check result of Logical AND when 2nd operand is a short vector.
    if ((int_vec.x != ((int_var - 1) && (int_var - 1))) || (int_vec.y != ((int_var - 2) && (int_var - 1))) ||
        (int_vec.z != ((int_var - 3) && (int_var - 1))))
        RET[programIndex] = -5;

    int_vec = int_vec1 && (int_var - 1);
    // Check result of Logical AND when 1st operand is a short vector.
    if ((int_vec.x != ((int_var - 1) && (int_var - 1))) || (int_vec.y != ((int_var - 2) && (int_var - 1))) ||
        (int_vec.z != ((int_var - 3) && (int_var - 1))))
        RET[programIndex] = -6;
}

task void result(uniform float ret[]) { ret[programIndex] = 0; }
