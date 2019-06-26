// RUN: not %{ispc} %s > %t 2>&1
// RUN: FileCheck --input-file=%t %s
// CHECK: Error:
// CHECK-NOT: Please file a bug
extern "C" int32 foo(int8* ptr) {
    int i = 0;
    while(*ptr != 0){
        i++;
    }
    return i;
}
