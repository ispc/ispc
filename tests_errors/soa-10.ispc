// Can't convert between types "uniform struct Foo" and "soa<4> struct Foo" with different SOA widths

struct Pt { float x, y, z; };

struct Foo { 
//CO    uniform int a; 
//CO    int a;
//CO    float *x;
//CO    float y[5], z; 
    double d[5];
    float x, y, z;
    //Pt p; 
};

extern void bar(float);

uniform Foo uf;
varying Foo vf;

int index;

export void x(uniform Foo &x, soa<8> Foo f[]) {
    soa<4> Foo xyz;
    xyz= uf;
}
