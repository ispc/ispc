
// RUN: %{ispc} --target=host --nowrap --emit-llvm-text -o - %s | FileCheck %s

struct Vec3 {
    float x, y, z;
};

// Unary negation operator
noinline Vec3 operator-(Vec3 v) {
    Vec3 result;
    result.x = -v.x;
    result.y = -v.y;
    result.z = -v.z;
    return result;
}

// Logical NOT operator
noinline bool operator!(Vec3 v) {
    return (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f);
}

// Prefix increment
noinline Vec3 operator++(Vec3 &v) {
    v.x += 1.0f;
    v.y += 1.0f;
    v.z += 1.0f;
    return v;
}

// Postfix increment
noinline Vec3 operator++(Vec3 &v, int) {
    Vec3 temp = v;
    v.x += 1.0f;
    v.y += 1.0f;
    v.z += 1.0f;
    return temp;
}

// CHECK-LABEL: test_unary_ops___
unmasked float test_unary_ops() {
    Vec3 v;
    v.x = 1.0f;
    v.y = 2.0f;
    v.z = 3.0f;

    // CHECK: call <{{[0-9]*}} x {{.*}}> @"operator!___s_5B_vyVec3_5D_"
    bool isZero = !v;

    if (isZero) {
        return 0.0f;
    }
    // CHECK: call %v{{[0-9]*}}_varying_Vec3 @operator-___s_5B_vyVec3_5D_
    Vec3 negated = -v;

    // CHECK: call %v{{[0-9]*}}_varying_Vec3 @"operator++___REFs_5B_vyVec3_5D_"
    Vec3 incremented = ++v;

    // CHECK: call %v{{[0-9]*}}_varying_Vec3 @"operator++___REFs_5B_vyVec3_5D_vyi"
    Vec3 pre_incremented = v++;

    return negated.x + negated.y + negated.z +
           incremented.x + incremented.y + incremented.z +
           pre_incremented.x + pre_incremented.y + pre_incremented.z;
}