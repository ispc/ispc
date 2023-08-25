// Tests template instantiation involving pointers/arrays with variability
// resolution.

// RUN: %{ispc}  %s --emit-llvm-text --target=host -o - | FileCheck %s

// CHECK: define void @foo
// CHECK: call void @aos_to_soa2_ispc___vyivyi___un_3C_uni_3E_un_3C_vyi_3E_un_3C_vyi_3E_REFCvyi({{(i32*|ptr)}}
// CHECK: call void @aos_to_soa2_ispc___vyfvyi___un_3C_unf_3E_un_3C_vyf_3E_un_3C_vyf_3E_REFCvyi({{(float*|ptr)}}

template <typename T, typename C>
noinline void aos_to_soa2_ispc(uniform T src[], varying T *uniform v0, varying T *uniform v1, const C &ind) {
    const varying T src0 = src[programIndex];
    const varying T src1 = src[programCount + programIndex];

    *v0 = shuffle(src0, src1, ind);
    *v1 = shuffle(src0, src1, ind);
}

void foo(uniform float srcf[], varying float *uniform v0f, varying float *uniform v1f, uniform int srci[],
         varying int *uniform v0i, varying int *uniform v1i, int ind) {
    aos_to_soa2_ispc<int, int>(srci, v0i, v1i, ind);
    aos_to_soa2_ispc<float, int>(srcf, v0f, v1f, ind);
}
