// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "attention_ispc.h" // Include the ISPC-generated header
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// Wrapper function for the optimized single head attention
void single_head_attention(nb::ndarray<float> Q, nb::ndarray<float> K, nb::ndarray<float> V, nb::ndarray<float> output,
                           int seq_len, int d_model) {

    // Call the ISPC function
    ispc::single_head_attention(Q.data(), K.data(), V.data(), output.data(), seq_len, d_model);
}

NB_MODULE(ispc_attention, m) {
    // Register the optimized_single_head_attention function
    m.def("single_head_attention", &single_head_attention, "Compute single head attention using ISPC", nb::arg("Q"),
          nb::arg("K"), nb::arg("V"), nb::arg("output"), nb::arg("seq_len"), nb::arg("d_model"));
}