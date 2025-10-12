// RUN: %{ispc} --target=host --nostdlib -o - --emit-llvm-text %s 2>&1 | FileCheck %s

struct Point { float x, y, z; };
struct SoaPoints {
    soa<8> Point pts[10];
};

extern bar(SoaPoints&);

// CHECK: %v{{[0-9]*}}_varying_SoaPoints = type { [10 x %v{{[0-9]*}}_soa8_Point] }
// CHECK: %v{{[0-9]*}}_soa8_Point = type { [{{[0-9]*}} x float], [{{[0-9]*}} x float], [{{[0-9]*}} x float] }
export void foo() {
    SoaPoints f;
    bar(f);
}
