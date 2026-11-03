/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

// Verify that print() flushes stdout immediately without flushing other streams,
// and that num_cores() agrees with the native libc.
// RUN: %{ispc} %S/3887-libc-linux.ispc --target=host --pic -o %t.o
// RUN: %{cxx} %s %t.o -o %t.bin
// RUN: %t.bin | FileCheck %s
// REQUIRES: LINUX_HOST && LINUX_ENABLED
// REQUIRES: (X86_64_HOST && X86_ENABLED) || (AARCH64_HOST && ARM_ENABLED)
// CHECK: C builtins runtime checks passed

#include <cassert>
#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>

extern "C" void foo(int *);

int main() {
    FILE *other = tmpfile();
    FILE *output = tmpfile();
    assert(other && output);
    static char otherBuffer[1024], outputBuffer[1024];
    assert(setvbuf(other, otherBuffer, _IOFBF, sizeof(otherBuffer)) == 0);
    assert(setvbuf(stdout, outputBuffer, _IOFBF, sizeof(outputBuffer)) == 0);
    assert(fputs("pending", other) >= 0);
    struct stat status;
    assert(fstat(fileno(other), &status) == 0 && status.st_size == 0);

    int savedStdout = dup(fileno(stdout));
    assert(savedStdout >= 0);
    assert(dup2(fileno(output), fileno(stdout)) >= 0);
    int cores = 0;
    foo(&cores);
    assert(cores == sysconf(_SC_NPROCESSORS_ONLN));
    assert(fstat(fileno(output), &status) == 0 && status.st_size > 0);
    assert(fstat(fileno(other), &status) == 0 && status.st_size == 0);

    assert(dup2(savedStdout, fileno(stdout)) >= 0);
    close(savedStdout);
    fclose(other);
    fclose(output);
    puts("C builtins runtime checks passed");
    fflush(stdout);
}
