// RUN: %{ispc} %s > %t 2>&1

struct Vec3f {
};

inline uniform float signedDist(uniform const Vec3f& pt) {
    return 0;
}

struct TriangleVerts
{
    Vec3f v0;
    Vec3f v1;
    Vec3f v2;
};

uniform TriangleVerts get_triangle_verts()
{
    uniform TriangleVerts tri;
    return tri;
}

void csClosestPoint()
{
    uniform TriangleVerts tri = get_triangle_verts();

    uniform const float d0 = signedDist(tri.v0);
}
