// AMX INT8 matrix multiply test
// This test verifies AMX INT8 dot product (tdpbssd) functionality.
// INT8 is supported on SPR and later targets.
//
// rule: skip on target=*
// rule: run on target=avx512spr.*
// rule: skip on arch=x86

#include "test_static.isph"
#include <amx.isph>

// Simple matmul test: C[2x1] = A[2x4] * B[4x1] using signed INT8
// For INT8 VNNI: A.colsb must equal B.rows * 4
// With K=4: A has 4 colsb, B has 1 row (K/4=1)
task void f_v(uniform float RET[]) {
    RET[programIndex] = 0;

    // Tile config: aligned to 64 bytes
    uniform int8 config[64];
    for (uniform int i = 0; i < 64; i++) config[i] = 0;

    // Set palette 1
    config[0] = 1;

    // Configure tiles for C[2x1] = A[2x4] * B[4x1]:
    // Tile 0 (C): 2 rows x 4 colsb (1 INT32 per row)
    // Tile 1 (A): 2 rows x 4 colsb (4 INT8 per row, K=4)
    // Tile 2 (B): 1 row x 4 colsb (VNNI format: K/4=1 row, N=1 so N*4=4 bytes)
    // Constraint: A.colsb (4) = B.rows (1) * 4

    // Column bytes (16-bit values at bytes 16-47)
    uniform int16 *uniform colsb = (uniform int16 *uniform)&config[16];
    colsb[0] = 4;   // tile 0: 1 int32 = 4 bytes
    colsb[1] = 4;   // tile 1: 4 int8 = 4 bytes (K=4)
    colsb[2] = 4;   // tile 2: N*4 = 4 bytes (N=1)

    // Rows (bytes 48-63)
    config[48] = 2;  // tile 0: M=2 rows
    config[49] = 2;  // tile 1: M=2 rows
    config[50] = 1;  // tile 2: K/4=1 row (VNNI packs 4 int8)

    // Load tile configuration
    amx_tile_loadconfig(config);

    // Matrix A: 2x4 INT8, all elements = 1
    uniform int8 A[8];  // 2 rows * 4 bytes
    for (uniform int i = 0; i < 8; i++) {
        A[i] = 1;
    }

    // Matrix B: 4x1 INT8 in VNNI format (1 row x 4 bytes)
    // VNNI packs 4 consecutive K values together
    uniform int8 B[4];  // 1 row * 4 bytes
    for (uniform int i = 0; i < 4; i++) {
        B[i] = 1;
    }

    // Matrix C: 2x1 INT32, output buffer
    uniform int8 C[8];  // 2 rows * 4 bytes
    for (uniform int i = 0; i < 8; i++) C[i] = 0;

    // Zero the destination tile
    amx_tile_zero(0);

    // Load A into tile 1 (stride = 4 bytes per row)
    amx_tile_load(1, A, 4);

    // Load B into tile 2 (stride = 4 bytes per row)
    amx_tile_load(2, B, 4);

    // Compute C = A * B using signed INT8 dot product
    amx_dpbssd(0, 1, 2);

    // Store result from tile 0
    amx_tile_store(0, C, 4);

    // Release tiles
    amx_tile_release();

    // Store computed result: C[0,0] = sum of A[0,k] * B[k,0] for k=0..3
    uniform int32 *uniform res = (uniform int32 *uniform)&C[0];
    RET[programIndex] = (float)res[0];
}

// Expected: C[i,0] = 4 * (1*1) = 4
task void result(uniform float RET[]) { RET[programIndex] = 4.0f; }
