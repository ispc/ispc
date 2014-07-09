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
#include <vector>
#include "timing.h"
#include "ispc_malloc.h"
#include "inplaceTranspose_ispc.h"
#include "typeT.h"

/* progress bar by Ross Hemsley;
 * http://www.rosshemsley.co.uk/2011/02/creating-a-progress-bar-in-c-or-any-other-console-app/ */
static inline void progressbar (unsigned int x, unsigned int n, unsigned int w = 50)
{
  if (n < 100)
  {
    x *= 100/n;
    n = 100;
  }

  if ((x != n) && (x % (n/100) != 0)) return;

  using namespace std;
  float ratio  =  x/(float)n;
  int c =  ratio * w;

  cerr << setw(3) << (int)(ratio*100) << "% [";
  for (int x=0; x<c; x++) cerr << "=";
  for (int x=c; x<w; x++) cerr << " ";
  cerr << "]\r" << flush;
}

int main(int argc, char * argv[])
{
  int m = argc > 1 ? atoi(argv[1]) : 8;
  int n = argc > 2 ? atoi(argv[2]) : 12;
  bool verbose = argc > 3;


  fprintf(stderr, " m= %d  n= %d :: storage= %g MB\n", m, n,
      m*n*sizeof(int)*2/1e6);


  std::pair<int,int> *A     = new std::pair<int,int>[m*n];
  std::pair<int,int> *Acopy = new std::pair<int,int>[m*n];

  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      A[j*m+i] = std::make_pair(i,j);

  if (verbose)
  {
    fprintf(stderr, "Original: \n");
    for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < m; i++)
      {
        fprintf(stderr, "(%2d,%2d) ", A[j*m+i].first, A[j*m+i].second);
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    for (int i = 0; i < m*n; i++)
      fprintf(stderr, "(%2d,%2d) ", A[i].first, A[i].second);
    fprintf(stderr, "\n");
    fprintf(stderr, "\n");
  }


  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      assert(A[j*m+i].first == i && A[j*m+i].second == j);

  ispcSetMallocHeapLimit(1024ull*1024*1024*8);
  ispcMemcpy(&Acopy[0], &A[0], sizeof(T)*m*n);

  int nrep = 10;
  double dt = 1e10;
  for (int r = 0; r < nrep; r++)
  {
    ispcMemcpy(&A[0], &Acopy[0], sizeof(T)*m*n);
    reset_and_start_timer();
    ispc::transpose((T*)&A[0], n, m);
    const double t1 = rtc();
    dt = std::min(dt, get_elapsed_msec());
    progressbar (r, nrep);
  }
  progressbar (nrep, nrep);
  fprintf(stderr, "\n");

  if (verbose)
  {
    fprintf(stderr, "Transposed: \n");
    for (int j = 0; j < m; j++)
    {
      for (int i = 0; i < n; i++)
      {
        fprintf(stderr, "(%2d,%2d) ", A[j*n+i].first, A[j*n+i].second);
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    for (int i = 0; i < m*n; i++)
      fprintf(stderr, "(%2d,%2d) ", A[i].first, A[i].second);
    fprintf(stderr, "\n");
    fprintf(stderr, "\n");
  }

  for (int j = 0; j < m; j++)
    for (int i = 0; i < n; i++)
      assert(A[j*n+i].first == j && A[j*n+i].second == i);

  fprintf(stderr, " tranpose done in %g msec :: BW= %g GB/s\n", 
      dt , 2*m*n*sizeof(int)*2/dt*1e3/1e9);


  return 0;
}
