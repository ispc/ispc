//; RUN: %{ispc} %s --target-os=windows --pic 2>&1 | FileCheck %s -check-prefix=CHECK_PIC
//; RUN: %{ispc} %s --target-os=linux --dllexport 2>&1 | FileCheck %s -check-prefix=CHECK_DLLEXPORT

//; CHECK_PIC: Warning: --pic switch for Windows target will be ignored.
//; CHECK_DLLEXPORT: Warning: --dllexport switch will be ignored, as the target OS is not Windows.
uniform int i;
