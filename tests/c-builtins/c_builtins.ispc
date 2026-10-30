/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

// Calls the C builtins, print() and num_cores(), see builtins/builtins-c-cpu.cpp.
// main.c runs it on the target OS and checks the results (#3887).

export uniform int c_builtins() {
    print("uniform: % % %\n", 42, 1.5f, 2.25d);
    print("varying: %\n", programIndex);
    return num_cores();
}
