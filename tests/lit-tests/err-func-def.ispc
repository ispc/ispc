// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Illegal "typedef" provided with function definition

typedef float foo(float a, float b) { }

// CHECK: Error: Function can't have both "task" and "export" qualifiers 

export task void bar() {
}

// CHECK: Error: Illegal to overload function by return type only 

float func() {
    int x = { 2 };
}

int y = { 2 };

void func() {
//CO    while (true)
//CO        ;
//CO    for (;;)
//CO        ;
    do ; while(1);
}

// CHECK: Error: Ignoring redefinition of function "redef".

float redef() {
    int x = { 2 };
}

int y = { 2 };

float redef() {
//CO    while (true)
//CO        ;
//CO    for (;;)
//CO        ;
    do ; while(1);
}
