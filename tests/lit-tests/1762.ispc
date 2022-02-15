// RUN: %{ispc} --target=avx512skx-x16 %s > %t 2>&1
// REQUIRES: X86_ENABLED
bool e[4][20][16];
void a() {
    for (int16 b = 0; b < 4; b++)
        for (int8 c = 0; c < 20; c++)
            foreach (d = 0...16)
                e[b][c][d] = 0;
}
