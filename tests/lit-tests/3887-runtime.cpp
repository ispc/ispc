/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

// Verify that print() flushes stdout immediately without flushing other streams,
// and that num_cores() agrees with the native libc.
// RUN: %{ispc} %S/3887-libc-linux.ispc --target=host --pic -o %t.o
// RUN: %{cxx} %s %t.o -o %t.bin
// RUN: %t.bin | FileCheck %s
// RUN: %{cxx} -DNDEBUG %s %t.o -o %t.ndebug.bin
// RUN: %t.ndebug.bin | FileCheck %s
// REQUIRES: LINUX_HOST && LINUX_ENABLED
// REQUIRES: (X86_64_HOST && X86_ENABLED) || (AARCH64_HOST && ARM_ENABLED)
// CHECK: C builtins runtime checks passed

#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>

extern "C" void foo(int *);

static void check(bool success, const char *message) {
    if (!success) {
        fprintf(stderr, "%s\n", message);
        exit(EXIT_FAILURE);
    }
}

int main() {
    FILE *other = tmpfile();
    FILE *output = tmpfile();
    check(other && output, "tmpfile failed");
    static char otherBuffer[1024], outputBuffer[1024];
    check(setvbuf(other, otherBuffer, _IOFBF, sizeof(otherBuffer)) == 0, "setvbuf(other) failed");
    check(setvbuf(stdout, outputBuffer, _IOFBF, sizeof(outputBuffer)) == 0, "setvbuf(stdout) failed");
    check(fputs("pending", other) >= 0, "fputs(other) failed");
    struct stat status;
    check(fstat(fileno(other), &status) == 0 && status.st_size == 0, "other stream was flushed");

    int savedStdout = dup(fileno(stdout));
    check(savedStdout >= 0, "dup(stdout) failed");
    check(dup2(fileno(output), fileno(stdout)) >= 0, "redirecting stdout failed");
    int cores = 0;
    foo(&cores);
    check(cores == sysconf(_SC_NPROCESSORS_ONLN), "num_cores() disagrees with libc");
    check(fstat(fileno(output), &status) == 0 && status.st_size > 0, "print() did not flush stdout");
    check(fstat(fileno(other), &status) == 0 && status.st_size == 0, "other stream was flushed");

    check(dup2(savedStdout, fileno(stdout)) >= 0, "restoring stdout failed");
    close(savedStdout);
    fclose(other);
    fclose(output);
    puts("C builtins runtime checks passed");
    fflush(stdout);
}
