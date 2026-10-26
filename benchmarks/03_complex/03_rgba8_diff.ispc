// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Marks the pixels of two RGBA8 images that differ (discussion #2943). Every pixel is 4 contiguous bytes, but the
// byte accesses at index * 4 + c are lowered as gathers and scatters.
export void difference_rgba8(const uniform uint8 source[], const uniform uint8 destination[], uniform uint8 output[],
                             uniform uint32 count) {
    foreach (index = 0 ... count) {
        if (source[index * 4 + 0] != destination[index * 4 + 0] ||
            source[index * 4 + 1] != destination[index * 4 + 1] ||
            source[index * 4 + 2] != destination[index * 4 + 2] ||
            source[index * 4 + 3] != destination[index * 4 + 3]) {
            output[index * 4 + 0] = 255;
            output[index * 4 + 1] = 0;
            output[index * 4 + 2] = 255;
            output[index * 4 + 3] = 255;
        }
    }
}

// Reference: the same with one uint32 per pixel (contiguous vector loads and a masked store).
export void difference_rgba8_u32(const uniform uint32 source[], const uniform uint32 destination[],
                                 uniform uint32 output[], uniform uint32 count) {
    foreach (index = 0 ... count) {
        if (source[index] != destination[index]) {
            output[index] = 0xffff00ff; // bytes 255, 0, 255, 255 in little-endian order
        }
    }
}
