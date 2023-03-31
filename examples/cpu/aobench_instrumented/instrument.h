/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef INSTRUMENT_H
#define INSTRUMENT_H 1

#include <stdint.h>

extern "C" {
void ISPCInstrument(const char *fn, const char *note, int line, uint64_t mask);
}

void ISPCPrintInstrument();

#endif // INSTRUMENT_H
