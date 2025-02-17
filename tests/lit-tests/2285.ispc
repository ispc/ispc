// RUN: %{ispc} -O2 %s -o - --target=host --emit-llvm-text --nowrap | FileCheck %s

enum material_type {
  DIFFUSE = 0,
};

struct sphere {
  material_type type;
};

// CHECK-NOT: FATAL ERROR:
export uniform float render(uniform sphere spheres[]) {
    // this the offensive line. If I add "uniform" to the type of current_sphere, compilation succeeds.
    sphere current_sphere = spheres[0];
    return 0.0f;
}

struct sphere x;

// CHECK-LABEL: @foo___un_3C_s_5B_unsphere_5D__3E_
// CHECK-DAG: {{%.*}} = shufflevector <[[WIDTH:.*]] x [[BASE_TYPE:i.*]]> {{%.*}}, <[[WIDTH]] x [[BASE_TYPE]]> {{poison|undef}}, <[[WIDTH]] x [[BASE_TYPE]]> zeroinitializer
void foo(uniform sphere spheres[]) { x = spheres[0]; }
