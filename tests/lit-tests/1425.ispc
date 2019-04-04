//; RUN: %{ispc} %s --no-pragma-once -h %t_no_pragma.h -o /dev/null
//; RUN: FileCheck --input-file=%t_no_pragma.h %s --implicit-check-not "#pragma once"
//; RUN: %{ispc} %s -h %t_with_pragma.h -o /dev/null
//; RUN: FileCheck --input-file=%t_with_pragma.h %s -check-prefix=CHECK_PRAGMA
//; CHECK_PRAGMA: #pragma once

typedef void (*foo_fn)(const uniform int x);

export void call_foo(void (* uniform foo)(const uniform int x), const uniform int x)
{
    foo(x);
}
