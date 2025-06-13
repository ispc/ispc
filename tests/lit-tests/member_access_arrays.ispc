// Test struct member access with array members on function call returns
// RUN: %{ispc} --target=host --nowrap --nostdlib %s --emit-asm -o - | FileCheck %s --allow-empty
// CHECK-NOT: Error

struct Matrix3x3 {
    float data[9];
};

struct Vertex {
    float position[3];
    float normal[3];
    float texCoord[2];
};

Matrix3x3 makeIdentityMatrix() {
    Matrix3x3 m;
    for (uniform int i = 0; i < 9; i++) {
        m.data[i] = (i % 4 == 0) ? 1.0 : 0.0;
    }
    return m;
}

Vertex makeVertex(float px, float py, float pz, float nx, float ny, float nz, float u, float v) {
    Vertex vert;
    vert.position[0] = px;
    vert.position[1] = py;
    vert.position[2] = pz;
    vert.normal[0] = nx;
    vert.normal[1] = ny;
    vert.normal[2] = nz;
    vert.texCoord[0] = u;
    vert.texCoord[1] = v;
    return vert;
}

void test_array_member_access() {
    // Test accessing array members from function call returns
    float first_element = makeIdentityMatrix().data[0];
    float diagonal_element = makeIdentityMatrix().data[4];
    float last_element = makeIdentityMatrix().data[8];
}

void test_vertex_array_members() {
    // Test accessing different array members
    float pos_x = makeVertex(1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.5, 0.5).position[0];
    float pos_y = makeVertex(4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.2, 0.8).position[1];
    float pos_z = makeVertex(7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0.9, 0.1).position[2];

    float norm_x = makeVertex(10.0, 11.0, 12.0, 0.707, 0.707, 0.0, 0.0, 1.0).normal[0];
    float norm_y = makeVertex(13.0, 14.0, 15.0, 0.0, 0.707, 0.707, 1.0, 0.0).normal[1];
    float norm_z = makeVertex(16.0, 17.0, 18.0, 0.707, 0.0, 0.707, 0.3, 0.7).normal[2];

    float tex_u = makeVertex(19.0, 20.0, 21.0, -1.0, 0.0, 0.0, 0.25, 0.75).texCoord[0];
    float tex_v = makeVertex(22.0, 23.0, 24.0, 0.0, -1.0, 0.0, 0.6, 0.4).texCoord[1];
}

void test_array_in_expressions() {
    // Test array member access in arithmetic expressions
    float dot_product = makeVertex(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0).position[0] *
                       makeVertex(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0).normal[0] +
                       makeVertex(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0).position[1] *
                       makeVertex(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0).normal[1] +
                       makeVertex(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0).position[2] *
                       makeVertex(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0).normal[2];

    // Test array indexing with computed indices
    float matrix_trace = makeIdentityMatrix().data[0] + makeIdentityMatrix().data[4] + makeIdentityMatrix().data[8];
}