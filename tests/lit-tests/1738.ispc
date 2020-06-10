// RUN: %{ispc} %s --target=host -h %t.h
// RUN: FileCheck --input-file=%t.h %s

// CHECK: struct vec3f_SOA16 vx;

struct vec3f {
    float x, y, z;
};

struct linear_space {
    vec3f vx;
    vec3f vy;
    vec3f vz;
};

export void generate_linear_spaces(soa<16> linear_space &spaces) {}
