#include "../test_static.isph"
typedef bool<3> bool3;


// a[] = { 1, 2, 3, ... };
task void f_f(uniform float RET[], uniform float a[]) {
    RET[programIndex] = 0;
    int int_var = ((int)a[programIndex]) % 3;

    bool var1 = int_var == 0;
    bool var2 = int_var == 1;
    bool var3 = int_var == 2;

    bool3 bool3_vec1 = {var1, var2, var2};
    bool3 bool3_vec2 = {var1, var3, var2};

    // Extract from vector and check.
    bool var = bool3_vec1.z;
    if (var != var2)
        RET[programIndex] = -1;

    // Insert value in vector and check.
    bool3_vec1.z = var3;
    if (bool3_vec1[2] != var3)
        RET[programIndex] = -1;

    // Logical OR
    bool3 bool3_vec = bool3_vec1 || bool3_vec2;
    // Check result of Logical OR when both operands are short vectors.
    if ((bool3_vec.x != (var1 || var1)) || (bool3_vec.y != (var2 || var3)) || (bool3_vec.z != (var3 || var2)))
        RET[programIndex] = -1;

    bool3_vec = var2 || bool3_vec1;
    // Check result of Logical OR when only one operand is a short vector.
    if ((bool3_vec.x != (var2 || var1)) || (bool3_vec.y != (var2 || var2)) || (bool3_vec.z != (var2 || var3)))
        RET[programIndex] = -1;

    // Logical AND
    bool3_vec = bool3_vec1 && bool3_vec2;
    // Check result of Logical AND when both operands are short vectors.
    if ((bool3_vec.x != (var1 && var1)) || (bool3_vec.y != (var2 && var3)) || (bool3_vec.z != (var3 && var2)))
        RET[programIndex] = -1;

    bool3_vec = var2 && bool3_vec1;
    // Check result of Logical AND when only one operand is a short vector.
    if ((bool3_vec.x != (var2 && var1)) || (bool3_vec.y != (var2 && var2)) || (bool3_vec.z != (var2 && var3)))
        RET[programIndex] = -1;

    // Bitwise OR
    bool3_vec = bool3_vec1 | bool3_vec2;
    // Check result of Bitwise OR when both operands are short vectors.
    if ((bool3_vec.x != (var1 | var1)) || (bool3_vec.y != (var2 | var3)) || (bool3_vec.z != (var3 | var2)))
        RET[programIndex] = -1;

    bool3_vec = var2 | bool3_vec1;
    // Check result of Bitwise OR when only one operand is a short vector.
    if ((bool3_vec.x != (var2 | var1)) || (bool3_vec.y != (var2 | var2)) || (bool3_vec.z != (var2 | var3)))
        RET[programIndex] = -1;

    // Bitwise AND
    bool3_vec = bool3_vec1 & bool3_vec2;
    // Check result of Bitwise AND when both operands are short vectors.
    if ((bool3_vec.x != (var1 & var1)) || (bool3_vec.y != (var2 & var3)) || (bool3_vec.z != (var3 & var2)))
        RET[programIndex] = -1;

    bool3_vec = var2 & bool3_vec1;
    // Check result of Bitwise AND when only one operand is a short vector.
    if ((bool3_vec.x != (var2 & var1)) || (bool3_vec.y != (var2 & var2)) || (bool3_vec.z != (var2 & var3)))
        RET[programIndex] = -1;

    // Comparison Operations
    bool3_vec = (bool3_vec1 == bool3_vec2);
    // Check result of == when both operands are short vectors.
    if ((bool3_vec.x != (var1 == var1)) || (bool3_vec.y != (var2 == var3)) || (bool3_vec.z != (var3 == var2)))
        RET[programIndex] = -1;

    bool3_vec = (bool3_vec1 != bool3_vec2);
    // Check result of != when both operands are short vectors.
    if ((bool3_vec.x != (var1 != var1)) || (bool3_vec.y != (var2 != var3)) || (bool3_vec.z != (var3 != var2)))
        RET[programIndex] = -1;
}

task void result(uniform float ret[]) { ret[programIndex] = 0; }
