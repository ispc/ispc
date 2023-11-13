// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Iteration domain type in "foreach_tiled" loop must be an atomic, pointer, or enum type

struct Point { float x, y; };

uniform int foo(Point p) {
    uniform int count = 0;
    foreach_unique (pt in p)
        ++count;
    return count;
}


