// RUN: %{ispc} --target=host --nostdlib --nowrap -o %t.o %s

export void foo(void (* uniform f)(int));
export void foo(void (* uniform f)(int))
{
}
