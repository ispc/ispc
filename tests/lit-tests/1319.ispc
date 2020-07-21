//; RUN: %{ispc} %s -h %t.h
//; RUN: FileCheck --input-file=%t.h %s

typedef void (*foo_fn)(const uniform int x);
//; CHECK: extern void {{[a-zA-Z0-9_]*}}{{[[:space:]]*}}call_foo(void ( * foo)(const int32_t x), const int32_t x)
export void call_foo(void (* uniform foo)(const uniform int x), const uniform int x)
{
    foo(x);
}
