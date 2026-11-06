// Generic targets do not implement the dot product builtins, so on 32-bit ARM CPUs with dotprod or i8mm the
// dot4add_*/dot2add_* functions must still use the fallbacks instead of calling the __not_supported stubs.

// RUN: %{ispc} %s -I %S --arch=arm --target=generic-i32x4 --target-os=linux --cpu=cortex-a55 \
// RUN:   --nowrap --emit-asm -o - | FileCheck %s --check-prefix=ASM --implicit-check-not=__not_supported
// RUN: %{ispc} %s -I %S --arch=arm --target=generic-i32x4 --target-os=linux --cpu=cortex-a55 \
// RUN:   --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=IR --implicit-check-not=__not_supported
// RUN: %{ispc} %s -I %S --arch=arm --target=generic-i32x4 --target-os=linux --cpu=cortex-a55 \
// RUN:   --nowrap -O0 --emit-asm -o - | FileCheck %s --check-prefix=ASM --implicit-check-not=__not_supported \
// RUN:   '--implicit-check-not={{__dot[24]add_}}'

// REQUIRES: ARM32_ENABLED && LINUX_ENABLED

// ASM: {{^}}dots___{{.*}}:
// IR: define {{.*}}@dots___

#include "generic-arm-dotprod.isph"
