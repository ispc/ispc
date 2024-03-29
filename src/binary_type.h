/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file dist.h
    @brief Defines function with distribution specific implementations.
*/

/** Print the binary type: slim or composite. */
void printBinaryType();

/** Initializes distribution specific paths based on main executable abspath. */
void initializeBinaryType(const char *MainExecutableAbsPath);
