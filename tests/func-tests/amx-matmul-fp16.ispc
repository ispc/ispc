// AMX FP16 matrix multiply test
// This test verifies AMX FP16 dot product (tdpfp16ps) functionality.
// FP16 is only supported on GNR and later targets.
//
// rule: skip on target=*
// rule: run on target=avx512gnr.*
// rule: skip on arch=x86

#include "test_static.isph"
#include <amx.isph>

// Simple matmul test: C[2x2] = A[2x4] * B[4x2] using FP16
// For FP16 VNNI: A.colsb must equal B.rows * 4
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
    // Tile 1 (A): 2 rows x 8 colsb (4 FP16 per row, K=4)
    // Tile 2 (B): 2 rows x 8 colsb (VNNI: K/2=2 rows, N*4=8 bytes)
    // Constraint: A.colsb (8) = B.rows (2) * 4

    // Column bytes (16-bit values at bytes 16-47)
    uniform int16 *uniform colsb = (uniform int16 *uniform)&config[16];
    colsb[0] = 8;   // tile 0: N*4 = 8 bytes (N=2 FP32)
    colsb[1] = 8;   // tile 1: K*2 = 8 bytes (K=4 FP16)
    colsb[2] = 8;   // tile 2: N*4 = 8 bytes (N=2)

    // Rows (bytes 48-63)
    config[48] = 2;  // tile 0: M=2 rows
    config[49] = 2;  // tile 1: M=2 rows
    config[50] = 2;  // tile 2: K/2=2 rows (VNNI packs 2 FP16)

    // Load tile configuration
    amx_tile_loadconfig(config);

    // Matrix A: 2x4 FP16, all elements = 1.0
    // FP16 1.0 = 0x3C00
    uniform int8 A[16];  // 2 rows * 8 bytes
    for (uniform int i = 0; i < 16; i += 2) {
        A[i] = 0x00;      // low byte of 0x3C00
        A[i + 1] = 0x3C;  // high byte of 0x3C00
    }

    // Matrix B: 4x2 FP16 in VNNI format (2 rows x 8 bytes)
    // VNNI packs pairs of K values: [B[0,0], B[1,0], B[0,1], B[1,1]] per row
    uniform int8 B[16];  // 2 rows * 8 bytes
    for (uniform int i = 0; i < 16; i += 2) {
        B[i] = 0x00;      // low byte of 0x3C00
        B[i + 1] = 0x3C;  // high byte of 0x3C00
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

    // Compute C = A * B using FP16 dot product
    amx_dpfp16ps(0, 1, 2);

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
