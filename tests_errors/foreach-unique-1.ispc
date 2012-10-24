// Iteration domain type in "foreach_tiled" loop must be an atomic, pointer, or enum type

struct Point { float x, y; };

uniform int foo(Point p) {
    uniform int count = 0;
    foreach_unique (pt in p)
        ++count;
    return count;
}


