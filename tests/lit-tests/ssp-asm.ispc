// Make sure the backend honors the request to generate code to detect stack smashing.
// We check for a call to  __security_check_cookie.
// RUN: %{ispc} -O0 --emit-asm %s --target=host --stack-protector=strong -o - | FileCheck %s
// RUN: %{ispc} -O1 --emit-asm %s --target=host --stack-protector=strong -o - | FileCheck %s
// RUN: %{ispc} -O2 --emit-asm %s --target=host --stack-protector=strong -o - | FileCheck %s
// RUN: %{ispc} -O3 --emit-asm %s --target=host --stack-protector=strong -o - | FileCheck %s

struct S {
    int arr[20];
};

//; CHECK-LABEL: test{{.*}}:
//; CHECK: {{(callq?|bl)}} {{(_+stack_chk_fail|_+security)}}

extern void foo(S *);

int test()
{
    uniform S s;
    foo(&s);
    return s.arr[14];
}
