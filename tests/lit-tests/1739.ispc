//; RUN: not %{ispc} --target=host %s --nowrap 2>&1 | FileCheck %s

//; CHECK: Error: Member operator "." can't be applied to pointer type "varying float<4>". Did you mean to use "->"?
float foo(float<4> foo_arg[4]) { return foo_arg.a; }
