typedef bool<4> bool4;


// a[] = { 1, 2, 3, ... };
export void f_f(uniform float RET[], uniform float a[]) {
    RET[programIndex] = 0;
    int int_var = ((int)a[programIndex]) % 4;

    bool var1 = int_var == 0;
    bool var2 = int_var == 1;
    bool var3 = int_var == 2;
    bool var4 = int_var == 3;

    bool4 bool4_vec1 = {var1, var2, var2, var4};
    bool4 bool4_vec2 = {var1, var3, var2, var4};

    // Extract from vector and check.
    bool var = bool4_vec1.z;
    if (var != var2)
        RET[programIndex] = -1;

    // Insert value in vector and check.
    bool4_vec1.z = var3;
    if (bool4_vec1[2] != var3)
        RET[programIndex] = -1;

    // Logical OR
    bool4 bool4_vec = bool4_vec1 || bool4_vec2;
    // Check result of Logical OR when both operands are short vectors.
    if ((bool4_vec.x != (var1 || var1)) || (bool4_vec.y != (var2 || var3)) || (bool4_vec.z != (var3 || var2)) ||
        (bool4_vec.w != (var4 || var4)))
        RET[programIndex] = -1;

    bool4_vec = var2 || bool4_vec1;
    // Check result of Logical OR when only one operand is a short vector.
    if ((bool4_vec.x != (var2 || var1)) || (bool4_vec.y != (var2 || var2)) || (bool4_vec.z != (var2 || var3)) ||
        (bool4_vec.w != (var2 || var4)))
        RET[programIndex] = -1;

    // Logical AND
    bool4_vec = bool4_vec1 && bool4_vec2;
    // Check result of Logical AND when both operands are short vectors.
    if ((bool4_vec.x != (var1 && var1)) || (bool4_vec.y != (var2 && var3)) || (bool4_vec.z != (var3 && var2)) ||
        (bool4_vec.w != (var4 && var4)))
        RET[programIndex] = -1;

    bool4_vec = var2 && bool4_vec1;
    // Check result of Logical AND when only one operand is a short vector.
    if ((bool4_vec.x != (var2 && var1)) || (bool4_vec.y != (var2 && var2)) || (bool4_vec.z != (var2 && var3)) ||
        (bool4_vec.w != (var2 && var4)))
        RET[programIndex] = -1;

    // Bitwise OR
    bool4_vec = bool4_vec1 | bool4_vec2;
    // Check result of Bitwise OR when both operands are short vectors.
    if ((bool4_vec.x != (var1 | var1)) || (bool4_vec.y != (var2 | var3)) || (bool4_vec.z != (var3 | var2)) ||
        (bool4_vec.w != (var4 | var4)))
        RET[programIndex] = -1;

    bool4_vec = var2 | bool4_vec1;
    // Check result of Bitwise OR when only one operand is a short vector.
    if ((bool4_vec.x != (var2 | var1)) || (bool4_vec.y != (var2 | var2)) || (bool4_vec.z != (var2 | var3)) ||
        (bool4_vec.w != (var2 | var4)))
        RET[programIndex] = -1;

    // Bitwise AND
    bool4_vec = bool4_vec1 & bool4_vec2;
    // Check result of Bitwise AND when both operands are short vectors.
    if ((bool4_vec.x != (var1 & var1)) || (bool4_vec.y != (var2 & var3)) || (bool4_vec.z != (var3 & var2)) ||
        (bool4_vec.w != (var4 & var4)))
        RET[programIndex] = -1;

    bool4_vec = var2 & bool4_vec1;
    // Check result of Bitwise AND when only one operand is a short vector.
    if ((bool4_vec.x != (var2 & var1)) || (bool4_vec.y != (var2 & var2)) || (bool4_vec.z != (var2 & var3)) ||
        (bool4_vec.w != (var2 & var4)))
        RET[programIndex] = -1;

    // Comparison Operations
    bool4_vec = (bool4_vec1 == bool4_vec2);
    // Check result of == when both operands are short vectors.
    if ((bool4_vec.x != (var1 == var1)) || (bool4_vec.y != (var2 == var3)) || (bool4_vec.z != (var3 == var2)) ||
        (bool4_vec.w != (var4 == var4)))
        RET[programIndex] = -1;

    bool4_vec = (bool4_vec1 != bool4_vec2);
    // Check result of != when both operands are short vectors.
    if ((bool4_vec.x != (var1 != var1)) || (bool4_vec.y != (var2 != var3)) || (bool4_vec.z != (var3 != var2)) ||
        (bool4_vec.w != (var4 != var4)))
        RET[programIndex] = -1;
}

export void result(uniform float ret[]) { ret[programIndex] = 0; }
