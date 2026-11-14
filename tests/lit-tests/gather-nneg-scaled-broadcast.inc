#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>

extern float read_scaled(void *, int32_t);

int main(void) {
    const size_t gib = (size_t)1 << 30;
    char *p = mmap(0, 6 * gib, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        perror("mmap");
        return 2;
    }
    *(float *)(p + gib) = -7.0f;
    *(float *)(p + 5 * gib) = 42.0f;
    float got = read_scaled(p + 3 * gib, 0x20000000);
    munmap(p, 6 * gib);
    if (got != 42.0f) {
        fprintf(stderr, "got %.0f, expected 42\n", got);
        return 1;
    }
    return 0;
}
