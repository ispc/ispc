// RUN: %{ispc} --target=generic-i1x64 --nostdlib %s --nowrap -o %t.o 2>&1 | FileCheck %s --allow-empty
// CHECK-NOT: Error: syntax error

// REQUIRES: X86_ENABLED && !MACOS_HOST

uniform int<128> data;