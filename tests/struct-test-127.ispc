struct Foo {
    uniform float x;
    varying float y;
};

inline varying Foo func(uniform Foo * varying f)
{
    return *f;
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Foo struct_1;
    struct_1.x = b;
    struct_1.y = aFOO[programIndex];

    uniform Foo * varying ptr = &(struct_1);
    varying Foo struct_2;
    struct_2.x = -100;
    struct_2.y = -150;
    if (programIndex % 3 == 0)
        struct_2 = func(ptr);
    RET[programIndex] = struct_2.x + struct_2.y;
}


export void result(uniform float RET[]) {
    if (programIndex % 3 == 0)
        RET[programIndex] = 5 + programIndex + 1;
    else
        RET[programIndex] = -145;
}

