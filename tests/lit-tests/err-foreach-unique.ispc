// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Iteration domain type in "foreach_tiled" loop must be an atomic, pointer, or enum type

struct Point { float x, y; };

uniform int foo(Point p) {
    uniform int count = 0;
    foreach_unique (pt in p)
        ++count;
    return count;
}

// CHECK: Error: Can't assign to type "const uniform int32"

void bar(int x) {
    foreach_unique (u in x)
        ++u;
}

// CHECK: Error: "break" statement is illegal outside of for/while/do loops

void func1(int x) {
    foreach_unique (u in x) {
        if (u == 0)
            break;
    }
}

// CHECK: Error: "return" statement is illegal inside a "foreach" loop

void func2(int x) {
    foreach_unique (u in x) {
        if (u == 0)
            return;
    }
}
