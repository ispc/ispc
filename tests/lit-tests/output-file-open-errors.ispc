// Copyright 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// A suffix warning must not replace the operating-system error from fopen.
// REQUIRES: LINUX_HOST || MACOS_HOST
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -h "%t.nonexistent/output.txt" 2>&1 | FileCheck %s
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap --dev-stub "%t.nonexistent/output.txt" 2>&1 | FileCheck %s
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap --host-stub "%t.nonexistent/output.txt" 2>&1 | FileCheck %s

// CHECK: Warning: Emitting {{.*}} file, but filename "{{.*}}/output.txt" has suffix "txt"?
// CHECK-NEXT: {{.*}}/output.txt: No such file or directory

export void f() {}
