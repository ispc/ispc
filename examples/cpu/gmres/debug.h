/*
  Copyright (c) 2012-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <cassert>

/**************************************************************\
| Macros
\**************************************************************/
#define DEBUG

#ifdef DEBUG
#define ASSERT(expr) assert(expr)
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define ASSERT(expr)
#define DEBUG_PRINT(...)
#endif

#endif
