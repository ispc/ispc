## Copyright 2023 Intel Corporation
## SPDX-License-Identifier: BSD-3-Clause

if (ISPCRT_BUILD_CPU)
    add_subdirectory(cpu)
endif()

if (ISPCRT_BUILD_GPU)
    add_subdirectory(gpu)
endif()
