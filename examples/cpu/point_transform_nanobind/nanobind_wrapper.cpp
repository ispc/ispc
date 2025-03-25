// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "point_transform_ispc.h" // Include the ISPC-generated header

namespace nb = nanobind;

// Wrapper function for the ISPC transform_points with direct buffer approach
void transform_points(nb::ndarray<float> points_x, nb::ndarray<float> points_y, nb::ndarray<float> result_x,
                      nb::ndarray<float> result_y, ispc::Transform transform, float strength) {

    const int count = static_cast<int>(points_x.shape(0));

    // Call the ISPC function
    ispc::transform_points(points_x.data(), points_y.data(), result_x.data(), result_y.data(), transform, strength,
                           count);
}

NB_MODULE(ispc_transform, m) {
    // Define the Transform struct for Python
    nb::class_<ispc::Transform>(m, "Transform")
        .def(nb::init<>())
        .def_rw("scale_x", &ispc::Transform::scale_x)
        .def_rw("scale_y", &ispc::Transform::scale_y)
        .def_rw("translate_x", &ispc::Transform::translate_x)
        .def_rw("translate_y", &ispc::Transform::translate_y)
        .def_rw("rotation", &ispc::Transform::rotation);

    // Register the direct buffer transform_points function
    m.def("transform_points", &transform_points, "Transform points using ISPC", nb::arg("points_x"),
          nb::arg("points_y"), nb::arg("result_x"), nb::arg("result_y"), nb::arg("transform"), nb::arg("strength"));
}