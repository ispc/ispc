// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: "break" statement is illegal outside of for/while/do loops

void foo(int x) {
    foreach_unique (u in x) {
        if (u == 0)
            break;
    }
}


