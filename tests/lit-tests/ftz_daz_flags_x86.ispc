// Tests checks that stmxcsr/ldmxcsr intrinsics set/restore FTZ/DAZ flags when --opt=reset-ftz-daz is passed.
// --opt=reset-ftz-daz affects external ISPC functions only.

// RUN: %{ispc} %s --target=avx2 --arch=x86 --nostdlib --emit-llvm-text --opt=reset-ftz-daz -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2 --arch=x86 --nostdlib --emit-llvm-text -o - | FileCheck --check-prefixes=CHECK_NO_FTZ_DAZ %s

// CHECK-LABEL: @test_ftz_daz___
// CHECK-NOT: stmxcsr
// CHECK-NOT: ldmxcsr
// CHECK-LABEL: @export_test_ftz_daz___
// CHECK-NOT: stmxcsr
// CHECK-NOT: ldmxcsr
// CHECK-LABEL: @export_void_test_ftz_daz___
// CHECK-NOT: stmxcsr
// CHECK-NOT: ldmxcsr
// CHECK-LABEL: @externC_test_ftz_daz()
// CHECK: stmxcsr
// CHECK-COUNT-2: ldmxcsr
// CHECK-LABEL: @export_test_ftz_daz(
// CHECK: stmxcsr
// CHECK-COUNT-2: ldmxcsr

// CHECK_NO_FTZ_DAZ-NOT: stmxcsr
// CHECK_NO_FTZ_DAZ-NOT: ldmxcsr

// macOS doesn't support 32 bit x86.
// REQUIRES: X86_ENABLED && !MACOS_HOST
uniform float test_ftz_daz(){
    uniform float x = 0x1p-149;
    uniform float y = 0x0.fffffep-126f;
    return y - x;
}

export uniform float export_test_ftz_daz(){
    uniform float x = 0x1p-149;
    uniform float y = 0x0.fffffep-126f;
    return y - x;
}

export void export_void_test_ftz_daz(uniform float res[]){
    uniform float x = 0x1p-149;
    uniform float y = 0x0.fffffep-126f;
    res[0] = y - x;
}

extern "C" uniform float externC_test_ftz_daz(){
    uniform float x = 0x1p-149;
    uniform float y = 0x0.fffffep-126f;
    if (programIndex % 2)
        return y + x;
    return y - x;
}
