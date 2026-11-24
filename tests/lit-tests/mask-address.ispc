// A varying bool is stored as one byte per program instance, which is also
// what sizeof gives for __mask. With a mask of one bit per program instance,
// the mask memory is smaller: a pointer to __mask pointed to it, so reading
// through the pointer read past it. The address of __mask, also taken by
// binding a reference to it, is now the address of a copy of the mask stored
// as a varying bool, which follows the mask: a pointer taken outside an if
// reads the mask inside the if.

// RUN: %{ispc} --pic --target=generic-i1x16 -O2 --woff -h %t.h %s -o %t.o
// RUN: %{cc} -O2 -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.bin
// RUN: %t.bin | FileCheck %s --check-prefix=EXEC
// RUN: %{ispc} --pic --target=generic-i1x16 -O0 --woff -h %t0.h %s -o %t0.o
// RUN: %{cc} -O2 -x c -c %s -o %t0.c.o --include %t0.h
// RUN: %{cc} %t0.o %t0.c.o -o %t0.bin
// RUN: %t0.bin | FileCheck %s --check-prefix=EXEC

// RUN: %{ispc} --target=avx512skx-x16 -O2 --woff --emit-llvm-text %s -o - | FileCheck %s --check-prefix=IR
// RUN: %{ispc} --target=avx512skx-x16 -O0 -g --woff --emit-llvm-text %s -o - | FileCheck %s --check-prefix=DBG

// EXEC: ok

// A function that only reads __mask has no copy left.
// IR-LABEL: define {{.*}}@mask_value
// IR-NOT: full_mask_storage
// IR: ret
// IR-LABEL: define {{.*}}@mask_address
// IR: %full_mask_storage = alloca <16 x i8>
// IR: call void @consume(ptr {{.*}}%full_mask_storage, i64 16)
// IR-LABEL: define {{.*}}@mask_reference
// IR: %full_mask_storage = alloca <16 x i8>
// IR: call void @consume(ptr {{.*}}%full_mask_storage, i64 16)
// IR-LABEL: define {{.*}}@mask_reference_argument
// IR: %full_mask_storage = alloca <16 x i8>
// IR: call void @consume_reference(ptr {{.*}}%full_mask_storage)

// The debug info of __mask describes the copy.
// DBG-LABEL: define {{.*}}@mask_address
// DBG: #dbg_declare(ptr %full_mask_storage, ![[MASK:[0-9]+]],
// DBG: ![[MASK]] = !DILocalVariable(name: "__mask"

// Like 3985-1.ispc, this assumes an x86-64 Linux host.
// REQUIRES: LINUX_HOST && X86_ENABLED && X86_64_HOST

#ifdef ISPC
extern "C" void consume(const varying bool *uniform p, uniform uint64 n);

export void mask_value(uniform int out[]) {
    if (programIndex & 1) {
        out[programIndex] = __mask ? 1 : 0;
    }
}

export void mask_address() { consume(&__mask, sizeof(__mask)); }

export void mask_reference() {
    const varying bool &r = __mask;
    consume(&r, sizeof(r));
}

extern "C" void consume_reference(const varying bool &m);
export void mask_reference_argument() { consume_reference(__mask); }

static int read_pointer(const varying bool *uniform p) { return *p ? 1 : 0; }
static int read_reference(const varying bool &b) { return b ? 1 : 0; }

export void mask_read(uniform int out[], uniform int64 size[]) {
    out[programIndex] = -1;
    const varying bool *uniform p = &__mask;
    if (programIndex & 1) {
        out[programIndex] = read_pointer(&__mask) + 10 * read_pointer(p) + 100 * read_reference(__mask);
    }
    size[0] = sizeof(__mask);
}
#else
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static int64_t consumed_size;
void consume_reference(const void *p) {}
static unsigned char consumed[64];
void consume(const void *p, uint64_t n) {
    consumed_size = (int64_t)n;
    if (n <= sizeof(consumed)) {
        memcpy(consumed, p, n);
    }
}
int main(void) {
    int out[64];
    int64_t size[1];
    mask_read(out, size);
    for (int i = 0; i < 16; i++) {
        int expected = (i & 1) ? 111 : -1;
        if (out[i] != expected) {
            printf("lane %d: %d, expected %d\n", i, out[i], expected);
            return 1;
        }
    }
    mask_address();
    if (size[0] != 16 || consumed_size != 16) {
        printf("sizeof(__mask) = %lld, consumed %lld bytes, expected 16\n", (long long)size[0],
               (long long)consumed_size);
        return 1;
    }
    for (int i = 0; i < 16; i++) {
        if (consumed[i] == 0) {
            printf("byte %d of __mask is 0\n", i);
            return 1;
        }
    }
    printf("ok\n");
    return 0;
}
#endif
