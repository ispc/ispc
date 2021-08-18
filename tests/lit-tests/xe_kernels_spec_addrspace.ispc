// This test checks calls of export and extern "C" functions on CPU and GPU based on close to real-life example.
// RUN: %{ispc} %s --target=gen9-x8 --emit-llvm-text --nowrap --no-discard-value-names -O0 -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK-XE %s

// RUN: %{ispc} %s --target=avx2 --emit-llvm-text --nowrap --no-discard-value-names -O0 -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=CHECK-CPU %s

// REQUIRES: XE_ENABLED
struct ISPCParams {
    int *img_buf;
    int img_size;
};

struct Context {
    float *t1;
    int x;
    int y;
};

struct RayHit {
    int ray;
    int hit;
    int geomID;
    int *data;
};

struct vec3f {
    float x;
    float y;
    float z;
};

inline void contextInit(uniform Context *uniform context) {
    context->x = 0;
    context->y = 1;
}

// CHECK-XE: define internal void @swap___{{.*}}
// CHECK-XE: define void @intersectISPC___{{.*}}(i1 %{{.*}}, %v8_uniform_Context* noalias %{{.*}}, %v8_uniform_RayHit* noalias %{{.*}}, float* noalias %{{.*}}, i8* noalias %{{.*}}, <8 x i1> %__mask)
// CHECK-XE-NOT: addrspacecast %v8_uniform_Context addrspace(4)* %{{.*}} to %v8_uniform_Context*
// CHECK-XE-NOT: addrspacecast %v8_uniform_RayHit addrspace(4)* %{{.*}} to %v8_uniform_RayHit*
// CHECK-XE-NOT: addrspacecast i8 addrspace(4)* %{{.*}} to i8*
// CHECK-XE: call void @swap___{{.*}}

// CHECK-XE: declare dso_local spir_func void @intersectExternal(%v8_uniform_Context addrspace(4)* noalias, %v8_uniform_RayHit addrspace(4)* noalias, float addrspace(4)* noalias, i8 addrspace(4)* noalias)

// CHECK-XE: define void @ispc_test___{{.*}}(<8 x float>* noalias %out, <8 x i1> %__mask)

// CHECK-XE: define dso_local spir_func void @intersectISPC(i1 %{{.*}}, %v8_uniform_Context addrspace(4)* noalias %{{.*}}, %v8_uniform_RayHit addrspace(4)* noalias %{{.*}}, float addrspace(4)* noalias %{{.*}}, i8 addrspace(4)* noalias %{{.*}})
// CHECK-XE: addrspacecast %v8_uniform_Context addrspace(4)* %{{.*}} to %v8_uniform_Context*
// CHECK-XE: addrspacecast %v8_uniform_RayHit addrspace(4)* %{{.*}} to %v8_uniform_RayHit*
// CHECK-XE: addrspacecast i8 addrspace(4)* %{{.*}} to i8*
// CHECK-XE: call void @swap___{{.*}}

// CHECK-XE: define dllexport spir_kernel void @kernel(i8 addrspace(1)*
// CHECK-XE: addrspacecast %v8_uniform_Context* %{{.*}} to %v8_uniform_Context addrspace(4)*
// CHECK-XE: addrspacecast %v8_uniform_RayHit* %{{.*}} to %v8_uniform_RayHit addrspace(4)*
// CHECK-XE: addrspacecast float* %{{.*}} to float addrspace(4)*
// CHECK-XE: addrspacecast i8* %{{.*}} to i8 addrspace(4)*
// CHECK-XE: call void @intersectExternal(%v8_uniform_Context addrspace(4)* %{{.*}}, %v8_uniform_RayHit addrspace(4)* %{{.*}}, float addrspace(4)* %{{.*}}, i8 addrspace(4)* %{{.*}})
// CHECK-XE: call void @intersectISPC___{{.*}}(i1 {{.*}}, %v8_uniform_Context* {{.*}}, %v8_uniform_RayHit* {{.*}}, float* {{.*}}, i8* {{.*}}, <8 x i1> {{.*}}

// CHECK-XE: define dso_local spir_func void @ispc_test(<8 x float> addrspace(4)* noalias %out)
// CHECK-XE: define dllexport spir_kernel void @ispc_kernel(i8 addrspace(1)*
// CHECK-XE: call void @ispc_test___{{.*}}(<8 x float>* {{.*}})

// CHECK-CPU: define void @swap___{{.*}}
// CHECK-CPU: define void @intersectISPC___{{.*}}(i1 %{{.*}}, %v8_uniform_Context* noalias %{{.*}}, %v8_uniform_RayHit* noalias %{{.*}}, float* noalias %{{.*}}, i8* noalias %{{.*}}, <8 x i32> %__mask)
// CHECK-CPU-NOT: addrspacecast %v8_uniform_Context* %{{.*}} to %v8_uniform_Context addrspace(4)*
// CHECK-CPU-NOT: addrspacecast %v8_uniform_RayHit* %{{.*}} to %v8_uniform_RayHit addrspace(4)*
// CHECK-CPU-NOT: addrspacecast i8 addrspace(4)* %{{.*}} to i8*
// CHECK-CPU: call void @swap___{{.*}}

// CHECK-CPU: declare void @intersectExternal(%v8_uniform_Context* noalias, %v8_uniform_RayHit* noalias, float* noalias, i8* noalias)

// CHECK-CPU: define void @kernel___{{.*}}
// CHECK-CPU-NOT: addrspacecast %v8_uniform_Context* %{{.*}} to %v8_uniform_Context addrspace(4)*
// CHECK-CPU-NOT: addrspacecast %v8_uniform_RayHit* %{{.*}} to %v8_uniform_RayHit addrspace(4)*
// CHECK-CPU-NOT: addrspacecast i8* %{{.*}} to i8 addrspace(4)*
// CHECK-CPU: call void @intersectExternal(%v8_uniform_Context* %{{.*}}, %v8_uniform_RayHit* %{{.*}}, float* %{{.*}}, i8* %{{.*}})
// CHECK-CPU: call void @intersectISPC___{{.*}}(i1 {{.*}}, %v8_uniform_Context* {{.*}}, %v8_uniform_RayHit* {{.*}}, float* {{.*}}, i8* {{.*}}, <8 x i32> {{.*}}

// CHECK-CPU: define void @ispc_test___{{.*}}(<8 x float>* noalias %out, <8 x i32> %__mask)
// CHECK-CPU: define void @ispc_kernel___{{.*}}
// CHECK-CPU: call void @ispc_test___{{.*}}(<8 x float>* {{.*}})

// CHECK-CPU: define void @intersectISPC(i1 %{{.*}}, %v8_uniform_Context* noalias %{{.*}}, %v8_uniform_RayHit* noalias %{{.*}}, float* noalias %{{.*}}, i8* noalias %{{.*}})
// CHECK-CPU-NOT: addrspacecast %v8_uniform_Context addrspace(4)* %{{.*}} to %v8_uniform_Context*
// CHECK-CPU-NOT: addrspacecast %v8_uniform_RayHit addrspace(4)* %{{.*}} to %v8_uniform_RayHit*
// CHECK-CPU-NOT: addrspacecast float addrspace(4)* %{{.*}} to float*
// CHECK-CPU-NOT: addrspacecast i8 addrspace(4)* %{{.*}} to i8*
// CHECK-CPU: call void @swap___{{.*}}

// CHECK-CPU: define void @ispc_test(<8 x float>* noalias %out)
// CHECK-CPU-NOT: define void @ispc_kernel(

noinline void swap(uniform bool doSwap, float &a_foo, float &b_foo) { a_foo = doSwap ? b_foo : a_foo; }

export noinline void intersectISPC(uniform bool test, uniform Context *uniform context, uniform RayHit *uniform rayhit,
                                            uniform float ret[], void* uniform params) {
    if (rayhit->ray != 0)
        rayhit->hit = context->t1[rayhit->geomID];
    context->y = rayhit->hit;
    float a1 = context->t1[programIndex] + ret[programIndex];
    float b1 = context->t1[programIndex + programCount];
    swap(test, a1, b1);
}

extern "C" void intersectExternal(uniform Context *uniform context, uniform RayHit *uniform rayhit,
                                                  uniform float ret[], void* uniform params);

task void kernel(void *uniform _params, uniform float ret[]) {
    ISPCParams *uniform params = (ISPCParams * uniform) _params;
    const uniform int y = taskIndex1;
    const uniform int x = taskIndex0;

    uniform Context context;
    contextInit(&context);

    uniform RayHit rayhit;

    intersectExternal(&context, &rayhit, ret, params);
    intersectISPC(false, &context, &rayhit, ret, params);

    if (rayhit.geomID != 300) {
        params->img_buf[x + y * params->img_size] = -1;
    }
}

varying float LoadFloat(varying float * uniform InParam, const varying int Offset, const bool bConstant) {
    return bConstant == true ? InParam[0] : InParam[Offset];
}

export noinline void ispc_test(varying float * uniform out) {
    vec3f coord = {out[0], out[1], out[2]};
    int offset = out[programIndex];
    out[programIndex] = LoadFloat(out, offset, (out[programIndex] > out[programIndex + programCount]));
}

task void ispc_kernel(uniform float ret[]) {
    ret[programIndex] = programCount;
    float test = ret[programIndex];
    ispc_test(&test);
}
