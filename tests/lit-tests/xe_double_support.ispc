// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-spirv --nowrap -o %t.spv
// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --device=skl --emit-spirv --nowrap -o %t.spv
// RUN: not %{ispc} %s --target=xelp-x16 --arch=xe64 --device=tgllp --emit-spirv --nowrap -o %t.spv 2>&1 | FileCheck %s -check-prefix=CHECK_DOUBLE_ERROR
// RUN: not %{ispc} %s --target=xelp-x16 --arch=xe64 --device=dg1 --emit-spirv --nowrap -o %t.spv 2>&1 | FileCheck %s -check-prefix=CHECK_DOUBLE_ERROR
// RUN: not %{ispc} %s --target=xehpg-x16 --arch=xe64 --device=dg2 --emit-spirv --nowrap -o %t.spv 2>&1 | FileCheck %s -check-prefix=CHECK_DOUBLE_ERROR

// REQUIRES: XE_ENABLED

// CHECK_DOUBLE_ERROR: Error: 'double' type is not supported by the target

export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    double a = aFOO[programIndex];
    RET[programIndex] = a / b;
}
