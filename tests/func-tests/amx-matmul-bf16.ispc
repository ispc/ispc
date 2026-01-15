// AMX BF16 matrix multiply test
// This test verifies AMX BF16 dot product (tdpbf16ps) functionality.
// BF16 is supported on SPR and later targets.
//
// rule: skip on target=*
// rule: run on target=avx512spr.*
// rule: skip on arch=x86

#include "test_static.isph"
#include <amx.isph>

// Simple matmul test: C[2x2] = A[2x4] * B[4x2] using BF16
// For BF16 VNNI: A.colsb must equal B.rows * 4
// With K=4 (8 bytes): B has K/2=2 rows
task void f_v(uniform float RET[]) {
    RET[programIndex] = 0;

    // Tile config: aligned to 64 bytes
    uniform int8 config[64];
    for (uniform int i = 0; i < 64; i++) config[i] = 0;

    // Set palette 1
    config[0] = 1;

    // Configure tiles for C[2x2] = A[2x4] * B[4x2]:
    // Tile 0 (C): 2 rows x 8 colsb (2 FP32 per row, N=2)
    // Tile 1 (A): 2 rows x 8 colsb (4 BF16 per row, K=4)
    // Tile 2 (B): 2 rows x 8 colsb (VNNI: K/2=2 rows, N*4=8 bytes)

    // Column bytes (16-bit values at bytes 16-47)
    uniform int16 *uniform colsb = (uniform int16 *uniform)&config[16];
    colsb[0] = 8;   // tile 0: N*4 = 8 bytes (N=2 FP32)
    colsb[1] = 8;   // tile 1: K*2 = 8 bytes (K=4 BF16)
    colsb[2] = 8;   // tile 2: N*4 = 8 bytes (N=2)

    // Rows (bytes 48-63)
    config[48] = 2;  // tile 0: M=2 rows
    config[49] = 2;  // tile 1: M=2 rows
    config[50] = 2;  // tile 2: K/2=2 rows (VNNI packs 2 BF16)

    // Load tile configuration
    amx_tile_loadconfig(config);

    // Matrix A: 2x4 BF16, all elements = 1.0
    // BF16 1.0 = 0x3F80 (high 16 bits of float 1.0)
    uniform int8 A[16];  // 2 rows * 8 bytes
    for (uniform int i = 0; i < 16; i += 2) {
        A[i] = 0x80;      // low byte of 0x3F80
        A[i + 1] = 0x3F;  // high byte of 0x3F80
    }

    // Matrix B: 4x2 BF16 in VNNI format (2 rows x 8 bytes)
    // VNNI packs pairs of K values: [B[0,0], B[1,0], B[0,1], B[1,1]] per row
    uniform int8 B[16];  // 2 rows * 8 bytes
    for (uniform int i = 0; i < 16; i += 2) {
        B[i] = 0x80;      // low byte of 0x3F80
        B[i + 1] = 0x3F;  // high byte of 0x3F80
    }

    // Matrix C: 2x2 FP32, output buffer
    uniform int8 C[16];  // 2 rows * 8 bytes
    for (uniform int i = 0; i < 16; i++) C[i] = 0;

    // Zero the destination tile
    amx_tile_zero(0);

    // Load A into tile 1 (stride = 8 bytes per row)
    amx_tile_load(1, A, 8);

    // Load B into tile 2 (stride = 8 bytes per row)
    amx_tile_load(2, B, 8);

    // Compute C = A * B using BF16 dot product
    amx_dpbf16ps(0, 1, 2);

    // Store result from tile 0
    amx_tile_store(0, C, 8);

    // Release tiles
    amx_tile_release();

    // Store computed result: C[0,0] = sum of A[0,k] * B[k,0] for k=0..3
    uniform float *uniform res = (uniform float *uniform)&C[0];
    RET[programIndex] = res[0];
}

// Expected: C[i,j] = 4 * (1.0*1.0) = 4.0
task void result(uniform float RET[]) { RET[programIndex] = 4.0f; }
