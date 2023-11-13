// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Ignoring redefinition of function "foo".

float foo() {
    int x = { 2 };
}

int y = { 2 };

float foo() {
//CO    while (true)
//CO        ;
//CO    for (;;)
//CO        ;
    do ; while(1);
}

