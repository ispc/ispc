// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Definition of variable "foo" conflicts with definition at

extern int foo;
float foo;

// CHECK: Error: Definition of variable "bar" conflicts with definition at 

extern int bar;
extern float bar;

// CHECK: Error: Definition of variable "gl_x" conflicts with definition at

int gl_x;
float gl_x;

// CHECK: Error: Redefinition of variable "gl_y" is illegal

int gl_y;
int gl_y;