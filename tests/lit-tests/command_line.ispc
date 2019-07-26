//; RUN: %{ispc} %s --target-os=linux --dllexport 2>&1 | FileCheck %s -check-prefix=CHECK_DLLEXPORT

//; CHECK_DLLEXPORT: Warning: --dllexport switch will be ignored, as the target OS is not Windows.
uniform int i;
