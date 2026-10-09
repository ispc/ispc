// Copyright 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// The fast-math rcp rewrite applies only to floating-point division, so integer
// division must not warn that rcp() is missing.
// RUN: %{ispc} %s --target=host --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --allow-empty

// CHECK-NOT: rcp(
uniform int div_uniform(uniform int a, uniform int b) { return a / b; }
int div_varying(int a, int b) { return a / b; }
