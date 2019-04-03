//; RUN: %{ispc} %s --no-pragma-once -h %t.h -o /dev/null
//; RUN: FileCheck --input-file=%t.h %s --implicit-check-not "#pragma once"
typedef void (*foo_fn)(const uniform int x);

export void call_foo(void (* uniform foo)(const uniform int x), const uniform int x)
{
    foo(x);
}
