// Check that linking the executable succeeds in the presence of a reference to the function
// that is called in the event of a stack violation. The symbol is guaranteed to be generated
// by --stack-protector=all.

// RUN: %{ispc} --target=host --stack-protector=all -h %t.h -o %t.o %s
// RUN: %{cc} -x c++ -c -o %t-cxx.o --include %t.h %s
// RUN: %{cc} -o %t.bin %t.o %t-cxx.o
// On x86 mac hosts clang seems to produce an arm executable, causing this
// test to fail. Until this is fixed, disable the test on all macs.
// REQUIRES: !MACOS_HOST

struct S {
    int arr[20];
};

#ifdef ISPC
export uniform int test()
{
    uniform S s;
    return s.arr[14];
}
#else
using namespace ispc;

int main()
{
    (void) test();
}
#endif /* ISPC */
