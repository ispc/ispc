// This test checks header files created with and without vectorcall on Windows. 
// RUN: %{ispc} %s -h %t1.h --target=host --target-os=windows
// RUN: FileCheck --input-file=%t1.h %s -check-prefix=CHECK1
// RUN: %{ispc} %s -h %t2.h --disable-vectorcall --target=host --target-os=windows
// RUN: FileCheck --input-file=%t2.h %s -check-prefix=CHECK2

// REQUIRES: X86_ENABLED

// REQUIRES: WINDOWS_ENABLED

// CHECK1: extern int32_t __vectorcall addVal(int32_t val1, int32_t val2);
// CHECK2: extern int32_t addVal(int32_t val1, int32_t val2);
export uniform int addVal( uniform int val1, uniform int val2) {
    return val1 + val2;
}
