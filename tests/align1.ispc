// rule: skip on arch=genx32
// rule: skip on arch=genx64
struct InnerUniform {
    uniform int i;
};

struct f_sz {
    uniform int type, temp[5];
    int k[3];
    InnerUniform u;
};



export void keep_struct_declared(varying f_sz * uniform s) {
}

export void result(uniform float RET[]) {
    RET[0] = sizeof(f_sz);
}
