#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>

extern int64_t read_split(void *, int32_t, int32_t);

int main(void) {
    const size_t gib = (size_t)1 << 30;
    char *p = mmap(0, 6 * gib, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        perror("mmap");
        return 2;
    }
    *(int64_t *)(p + gib) = -7;
    *(int64_t *)(p + 5 * gib) = 42;
    int64_t got = read_split(p + 5 * gib, INT32_MAX, 1);
    munmap(p, 6 * gib);
    if (got != 42) {
        fprintf(stderr, "got %lld, expected 42\n", (long long)got);
        return 1;
    }
    return 0;
}
