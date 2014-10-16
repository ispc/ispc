/*
  Copyright (c) 2010-2014, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <iomanip>
#include "timing.h"
#include "ispc_malloc.h"
#include "radixSort_ispc.h"

static void progressBar(const int x, const int n, const int width = 50)
{
  assert(n > 1);
  assert(x >= 0 && x < n);
  assert(width > 10);
  const float f = static_cast<float>(x)/(n-1);
  const int   w = static_cast<int>(f * width);

  // print bar
  std::string bstr("[");
  for (int i = 0; i < width; i++)
    bstr += i < w ? '=' : ' ';
  bstr += "]";

  // print percentage 
  char pstr0[32];
  sprintf(pstr0, " %2d %c ", static_cast<int>(f*100.0),'%');
  const std::string pstr(pstr0);
  std::copy(pstr.begin(), pstr.end(), bstr.begin() + (width/2-2));

  std::cout << bstr;
  std::cout << (x == n-1 ? "\n" : "\r") << std::flush;
}

struct Key
{
  int32_t key,val;
};

int main (int argc, char *argv[])
{
  int i, j, n = argc == 1 ? 1000000 : atoi(argv[1]), m = n < 100 ? 1 : 50, l = n < 100 ? n : RAND_MAX;
  double tISPC1 = 0.0, tISPC2 = 0.0, tSerial = 0.0;
  Key *keys = new Key [n];
  Key *keys_orig = new Key [n];
  unsigned int *keys_gold = new unsigned int [n];

  srand48(rtc()*65536);

  int sortBits = 32;
  assert(sortBits <= 32);

#pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    keys[i].key = ((int)(drand48() * (1<<30))) & ((1ULL << sortBits) - 1);
    keys[i].val = i;
  }

  std::random_shuffle(keys, keys + n);

#pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    keys_gold[i] = keys[i].key;
    keys_orig[i] = keys[i];
  }

  ispcSetMallocHeapLimit(1024*1024*1024);

  ispc::radixSort_alloc(n);

  tISPC2 = 1e30;
  for (i = 0; i < m; i ++)
  {
    ispcMemcpy(keys, keys_orig, n*sizeof(Key));
    reset_and_start_timer();
    ispc::radixSort(n, (int64_t*)keys, sortBits);
    tISPC2 = std::min(tISPC2, get_elapsed_msec());
    if (argc != 3)
        progressBar (i, m);
  }

  ispc::radixSort_free();

  printf("[sort ispc + tasks]:\t[%.3f] msec [%.3f Mpair/s]\n", tISPC2, 1.0e-3*n/tISPC2);

  std::sort(keys_gold, keys_gold + n);
  for (int i = 0; i < n; i++)
    assert(keys[i].key == keys_gold[i]);


#if 0
  for (i = 0; i < m; i ++)
  {
    ispcMemcpy(code, code_orig, n*sizeof(unsigned int));

    reset_and_start_timer();

    sort_serial (n, code, order);

    tSerial += get_elapsed_msec();

    if (argc != 3)
        progressBar (i, m);
  }

  printf("[sort serial]:\t\t[%.3f] msec [%.3f Mpair/s]\n", tSerial, 1.0e-3*n*m/tSerial);

#ifndef _CUDA_
  printf("\t\t\t\t(%.2fx speedup from ISPC, %.2fx speedup from ISPC + tasks)\n", tSerial/tISPC1, tSerial/tISPC2);
#else
  printf("\t\t\t\t(%.2fx speedup from ISPC + tasks)\n", tSerial/tISPC2);
#endif
#endif

  delete keys;
  delete keys_orig;
  delete keys_gold;
  return 0;
}
