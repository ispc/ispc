struct Foo {
    uniform float x;
    varying float y;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Foo struct_1;
    struct_1.x = b;
    struct_1.y = aFOO[programIndex];
    varying Foo struct_2 = struct_1;
    RET[programIndex] = struct_2.x + struct_2.y;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 5 + programIndex + 1;
}

