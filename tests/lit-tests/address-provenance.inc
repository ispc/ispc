// Shared kernels/driver for the #3983 integer-to-varying-pointer cases.
// The caller supplies the integer distance between two actual objects.
// Ordinary out-of-bounds array indexing is not used to reach the second one.

#ifdef ISPC
uniform float origin[64];
uniform float destination[64];

// A same-block reload after a masked store. Numeric base recovery must not
// turn the integer-built pointer into a GEP based solely on origin.
export void same_block(uniform int relative, uniform int n, uniform float out[]) {
    float x = destination[programIndex];
    uniform float *varying p =
        (uniform float *varying)((uniform int64)&origin[0] + (int64)(relative + programIndex) * 4);
    if (programIndex < n)
        *p = 5.0f;
    out[programIndex] = x + destination[programIndex];
}

// The corresponding reload across the foreach/masked-store control flow.
// This is a gate for future post-inline cleanup, including late EarlyCSE.
export void across_blocks(uniform int relative, uniform int n, uniform float out[]) {
    foreach (i = 0 ... n) {
        float x = destination[i];
        uniform float *varying p =
            (uniform float *varying)((uniform int64)&origin[0] + (int64)(relative + i) * 4);
        if (x > 0)
            *p = 5.0f;
        out[i] = x + destination[i];
    }
}

export uniform int width() { return programCount; }
#else
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern float origin[64];
extern float destination[64];

static int check(int across, int n, int w) {
    float out[64];
    for (int i = 0; i < 64; ++i) {
        origin[i] = 100.0f + i;
        destination[i] = 1.0f;
        out[i] = -1.0f;
    }
    // Subtract integer addresses, not pointers to different arrays.
    int64_t bytes = (int64_t)((uintptr_t)destination - (uintptr_t)origin);
    int32_t relative = (int32_t)(bytes / 4);
    if (bytes % 4 || (int64_t)relative * 4 != bytes)
        return 1;
    if (across)
        across_blocks(relative, n, out);
    else
        same_block(relative, n, out);
    for (int i = 0; i < 64; ++i) {
        float expected = i < n ? 6.0f : !across && i < w ? 2.0f : -1.0f;
        if (out[i] != expected || destination[i] != (i < n ? 5.0f : 1.0f) || origin[i] != 100.0f + i) {
            printf("%s n=%d lane=%d: got %g, expected %g\n", across ? "across_blocks" : "same_block", n, i,
                   out[i], expected);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    int w = width();
    if (w < 2 || w > 64 || argc != 2)
        return 1;
    int full = !strcmp(argv[1], "full_block");
    int off = !strcmp(argv[1], "same_block_off");
    int across = full || !strcmp(argv[1], "across_blocks");
    if (!across && !off && strcmp(argv[1], "same_block"))
        return 1;
    // Keep the cases disjoint, so that a failing gate hides no passing one:
    // same_block_off is the all-off mask (n = 0), same_block runs n = 1..w,
    // across_blocks runs the partial foreach iterations n = 0..w-1 (n = 0 runs
    // no iteration), and full_block runs the full iteration n = w. same_block
    // and full_block already miscompile on AVX2.
    int first = full ? w : (across || off) ? 0 : 1;
    int last = off ? 0 : (across && !full) ? w - 1 : w;
    for (int n = first; n <= last; ++n)
        if (check(across, n, w))
            return 1;
    return 0;
}
#endif
