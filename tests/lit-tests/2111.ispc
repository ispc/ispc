//; RUN: %{ispc} --emit-asm %s --target=avx2-i32x8 --x86-asm-syntax=att --arch=x86-64 -o - | FileCheck %s

// REQUIRES: X86_ENABLED

struct FVector3 {
    float x, y, z;
};

static const varying int vAOS32 = {0, 1, 2, 3, 8, 9, 10, 11};

//; CHECK-LABEL: test:
//; CHECK: {{[vpermpd $80]|[vperm2f128]}}
//; CHECK-NOT: vpermpd $136
export void test(uniform FVector3 &Out, uniform float *uniform In)
{
    varying float vx;

    const varying float src0 = In[programIndex];
    const varying float src1 = In[programCount + programIndex];

    vx = shuffle(src0, src1, vAOS32);

    Out.x = reduce_add(vx);
}
