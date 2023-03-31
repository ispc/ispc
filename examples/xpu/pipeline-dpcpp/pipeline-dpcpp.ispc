/*
  Copyright (c) 2020-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

task void stage1(float *uniform mem, uniform int count) {
    foreach (index = 0 ... count) {
        // Load the appropriate input value for this program instance.
        float v = mem[index];
        // Modify it
        v += 1.0f;
        // Write the result back.
        mem[index] = v;
    }
}

task void stage3(float *uniform mem, uniform int count) {
    foreach (index = 0 ... count) {
        // Load the appropriate input value for this program instance.
        float v = mem[index];
        // Modify it
        v -= 0.5f;
        // Write the result back.
        mem[index] = v;
    }
}
