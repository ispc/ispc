/*
  Copyright (c) 2012-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <cassert>
#include <cstdio>

/**************************************************************\
| Macros
\**************************************************************/
#define DEBUG

// Define the correct format specifier for size_t
#ifdef _MSC_VER
  // MSVC: use '%zu'
  #define SIZE_T_FORMAT "%zu"
#else
  // Linux/macOS: use '%lu'
  #define SIZE_T_FORMAT "%lu"
#endif

#ifdef DEBUG
#define ASSERT(expr) assert(expr)
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define ASSERT(expr)
#define DEBUG_PRINT(...)
#endif

#endif
