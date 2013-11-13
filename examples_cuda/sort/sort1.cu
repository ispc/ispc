/*
  Copyright (c) 2013, Durham University
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Durham University nor the names of its
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

/* Author: Tomasz Koziara */

#define programCount 32
#define programIndex (threadIdx.x & 31)
#define taskIndex (blockIdx.x*4 + (threadIdx.x >> 5))
#define taskCount (gridDim.x*4)
#define cfor for
#define cif if

#define int8 char
#define int64 long
#define sync cudaDeviceSynchronize();

__device__ inline int nbx(const int n) { return (n - 1) / 4 + 1; }

__global__ void histogram ( int span,  int n,  int64 code[],  int pass,  int hist[])
{
  if (taskIndex >= taskCount) return;
   int start = taskIndex*span;
   int end = taskIndex == taskCount-1 ? n : start+span;
   int strip = (end-start)/programCount;
   int tail = (end-start)%programCount;
  int i = programCount*taskIndex + programIndex;
  int g [256];

  cfor (int j = 0; j < 256; j ++)
  {
    g[j] = 0;
  }

  cfor (int k = start+programIndex*strip; k < start+(programIndex+1)*strip; k ++)
  {
    unsigned int8 *c = (unsigned int8*) &code[k];

    g[c[pass]] ++;
  }

  if (programIndex == programCount-1) /* remainder is processed by the last lane */
  {
    for (int k = start+programCount*strip; k < start+programCount*strip+tail; k ++)
    {
      unsigned int8 *c = (unsigned int8*) &code[k];

      g[c[pass]] ++;
    }
  }

  cfor (int j = 0; j < 256; j ++)
  {
    hist[j*programCount*taskCount+i] = g[j];
  }
}

__global__ void permutation ( int span,  int n,  int64 code[],  int pass,  int hist[],  int64 perm[])
{
  if (taskIndex >= taskCount) return;
   int start = taskIndex*span;
   int end = taskIndex == taskCount-1 ? n : start+span;
   int strip = (end-start)/programCount;
   int tail = (end-start)%programCount;
  int i = programCount*taskIndex + programIndex;
  int g [256];

  cfor (int j = 0; j < 256; j ++)
  {
    g[j] = hist[j*programCount*taskCount+i];
  }

  cfor (int k = start+programIndex*strip; k < start+(programIndex+1)*strip; k ++)
  {
    unsigned int8 *c = (unsigned int8*) &code[k];

    int l = g[c[pass]];

    perm[l] = code[k];

    g[c[pass]] = l+1;
  }

  if (programIndex == programCount-1) /* remainder is processed by the last lane */
  {
    for (int k = start+programCount*strip; k < start+programCount*strip+tail; k ++)
    {
      unsigned int8 *c = (unsigned int8*) &code[k];

      int l = g[c[pass]];

      perm[l] = code[k];

      g[c[pass]] = l+1;
    }
  }
}

__global__ void copy ( int span,  int n,  int64 from[],  int64 to[])
{
  if (taskIndex >= taskCount) return;
   int start = taskIndex*span;
   int end = taskIndex == taskCount-1 ? n : start+span;

  for (int i = programIndex + start; i < end; i += programCount)
    if (i < end)
  {
    to[i] = from[i];
  }
}

__global__ void pack ( int span,  int n,  unsigned int code[],  int64 pair[])
{
  if (taskIndex >= taskCount) return;
   int start = taskIndex*span;
   int end = taskIndex == taskCount-1 ? n : start+span;

  for (int i = programIndex + start; i < end; i += programCount)
    if (i < end)
  {
    pair[i] = ((int64)i<<32)+code[i];
  }
}

__global__ void unpack ( int span,  int n,  int64 pair[],  int unsigned code[],  int order[])
{
  if (taskIndex >= taskCount) return;
   int start = taskIndex*span;
   int end = taskIndex == taskCount-1 ? n : start+span;

  for (int i = programIndex + start; i < end; i += programCount)
    if (i < end)
  {
    code[i] = pair[i];
    order[i] = pair[i]>>32;
  }
}

__global__ void addup ( int h[],  int g[])
{
  if (taskIndex >= taskCount) return;
   int *  u = &h[256*programCount*taskIndex];
   int i, x, y = 0;

  for (i = 0; i < 256*programCount; i ++)
  {
    x = u[i];
    u[i] = y;
    y += x;
  }

  g[taskIndex] = y;
}

__global__ void bumpup ( int h[],  int g[])
{
  if (taskIndex >= taskCount) return;
   int *  u = &h[256*programCount*taskIndex];
   int z = g[taskIndex];

  for (int i = programIndex; i < 256*programCount; i += programCount)
  {
    u[i] += z;
  }
}

__device__
static void prefix_sum ( int num,  int h[])
{
   int *  g =  new  int [num+1];
   int i;

//  launch[num] addup (h, g+1);
   if(programIndex == 0)
     addup<<<nbx(num),128>>>(h,g+1);
   sync;

  for (g[0] = 0, i = 1; i < num; i ++) g[i] += g[i-1];

//  launch[num] bumpup (h, g);
  if(programIndex == 0)
    bumpup<<<nbx(num),128>>>(h,g);
  sync;

  delete g;
}

extern "C" __global__
void sort_ispc ( int n,  unsigned int code[],  int order[],  int ntasks)
{
   int num = ntasks < 1 ? 13*4*8 : ntasks;
   int span = n / num;
   int hsize = 256*programCount*num;
   int *  hist =  new  int [hsize];
   int64 *  pair =  new  int64 [n];
   int64 *  temp =  new  int64 [n];
   int pass, i;


//  launch[num] pack (span, n, code, pair);
   if(programIndex == 0)
     pack<<<nbx(num),128>>>(span, n, code, pair);
  sync;

#if 0
  for (pass = 0; pass < 4; pass ++)
  {
//    launch[num] histogram (span, n, pair, pass, hist);
   if(programIndex == 0)
    histogram<<<nbx(num),128>>>(span, n, pair, pass, hist);
    sync;

    prefix_sum (num, hist);

//    launch[num] permutation (span, n, pair, pass, hist, temp);
   if(programIndex == 0)
    permutation<<<nbx(num),128>>> (span, n, pair, pass, hist, temp);
    sync;

///    launch[num] copy (span, n, temp, pair);
   if(programIndex == 0)
    copy<<<nbx(num),128>>> (span, n, temp, pair);
    sync;
  }

///  launch[num] unpack (span, n, pair, code, order);
   if(programIndex == 0)
  unpack<<<nbx(num),128>>> (span, n, pair, code, order);
  sync;
#endif


  delete hist;
  delete pair;
  delete temp;
}
