// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Define a structure to represent a 2D transformation
struct Transform {
    float scale_x;
    float scale_y;
    float translate_x;
    float translate_y;
    float rotation; // in radians
};

// Apply transformation to an array of 2D points
export void transform_points(uniform float points_x[], uniform float points_y[], uniform float result_x[],
                             uniform float result_y[], uniform Transform &transform, uniform float strength,
                             uniform int count) {
    uniform float sin_theta = sin(transform.rotation);
    uniform float cos_theta = cos(transform.rotation);

    foreach (i = 0 ... count) {
        // Apply scaling
        float x = points_x[i] * transform.scale_x;
        float y = points_y[i] * transform.scale_y;

        // Apply rotation
        float x_rot = x * cos_theta - y * sin_theta;
        float y_rot = x * sin_theta + y * cos_theta;

        // Apply translation and strength factor
        result_x[i] = x_rot + transform.translate_x * strength;
        result_y[i] = y_rot + transform.translate_y * strength;
    }
}