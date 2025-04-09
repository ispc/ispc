// RUN: %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s --allow-empty

// CHECK-NOT: Error: Assertion failed
struct FloatVec3 {
    float coords[3];
};

struct Triangle {
    FloatVec3 point[3];
};

struct MeshObject {
    unsigned int numTris;
    Triangle *triangles;
};

bool intersects_triangle_ispc(Triangle triangle);

export void intersects_mesh(uniform MeshObject const &MO, uniform float tOut[]) {
    foreach (i = 0...MO.numTris) {
        if (!intersects_triangle_ispc(MO.triangles[i])) {
            tOut[i] = -1;
        }
    }
}
