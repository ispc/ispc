// Regression test for https://github.com/ispc/ispc/issues/375
// A member pointer to a forward-declared struct can be used once the struct is
// defined, even with another struct defined in between.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.o

struct Primitive;

struct Geom {
    Primitive *prim;
};

struct GeomTreeNode {
    int x;
};

struct Primitive {
    int x;
};

unmasked void Init(GeomTreeNode *uniform node, Geom *uniform geom) { node->x = geom->prim->x; }
