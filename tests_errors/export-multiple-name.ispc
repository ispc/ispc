// Illegal to provide "export" qualifier for functions with the same name but different types

export void foo() { }

export void foo(uniform int x) { }
