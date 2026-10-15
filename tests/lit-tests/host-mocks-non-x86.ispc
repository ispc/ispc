// On non-x86 hosts --mock-host-target and --mock-host-os-apx are ignored with a
// warning, which --werror turns into a failure.

// RUN: %{ispc} %s --mock-host-target=avx2-i32x8 --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=TARGET-WARN
// RUN: not %{ispc} %s --mock-host-target=avx2-i32x8 --werror --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=TARGET-ERR
// RUN: %{ispc} %s --mock-host-os-apx=off --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=OS-APX-WARN
// RUN: not %{ispc} %s --mock-host-os-apx=on --werror --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=OS-APX-ERR

// REQUIRES: AARCH64_HOST || RISCV64_HOST || PPC64LE_HOST

// TARGET-WARN: Warning: --mock-host-target is ignored on non-x86 hosts.
// TARGET-ERR: Error: --mock-host-target is ignored on non-x86 hosts.
// OS-APX-WARN: Warning: --mock-host-os-apx is ignored on non-x86 hosts.
// OS-APX-ERR: Error: --mock-host-os-apx is ignored on non-x86 hosts.

uniform int foo() { return 0; }
