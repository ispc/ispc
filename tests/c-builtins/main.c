/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

// Runs c_builtins.ispc and checks that num_cores() is the number of online
// processors that the C library of the target OS reports. run.sh checks the
// output of print().

#include <stdio.h>
#include <unistd.h>

int c_builtins(void);

int main(void) {
    int cores = c_builtins();
    long expected = sysconf(_SC_NPROCESSORS_ONLN);
    printf("num_cores: %d, sysconf(_SC_NPROCESSORS_ONLN): %ld\n", cores, expected);
    return cores > 0 && cores == expected ? 0 : 1;
}
