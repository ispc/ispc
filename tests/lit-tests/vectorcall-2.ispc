// This test checks header files created with and without vectorcall on Windows.
// Enabled
// RUN: %{ispc} %s -h %t1.h --vectorcall --target=host --target-os=windows
// RUN: FileCheck --input-file=%t1.h %s -check-prefix=CHECK_ENABLED
// Disabled
// RUN: %{ispc} %s -h %t2.h --no-vectorcall --target=host --target-os=windows
// RUN: FileCheck --input-file=%t2.h %s -check-prefix=CHECK_DISABLED
// Default
// RUN: %{ispc} %s -h %t3.h --target=host --target-os=windows
// RUN: FileCheck --input-file=%t3.h %s -check-prefix=CHECK_DISABLED

// REQUIRES: X86_ENABLED

// REQUIRES: WINDOWS_ENABLED

// CHECK_ENABLED: extern int32_t __vectorcall addVal(int32_t val1, int32_t val2);
// CHECK_DISABLED: extern int32_t addVal(int32_t val1, int32_t val2);
export uniform int addVal( uniform int val1, uniform int val2) {
    return val1 + val2;
}
