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

#ifdef __NVPTX__
#warning "emitting DEVICE code"
#define programCount warpSize()
#define programIndex laneIndex()
#define taskIndex     blockIndex0()
#define taskCount     blockCount0()
#define cfor          for
#define cif           if
#else
#warning "emitting HOST code"
#endif

task void histogram (uniform int span, uniform int n, uniform int64 code[], uniform int pass, uniform int hist[])
{
  if (taskIndex >= taskCount) return;
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;
  uniform int strip = (end-start)/programCount;
  uniform int tail = (end-start)%programCount;
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

task void permutation (uniform int span, uniform int n, uniform int64 code[], uniform int pass, uniform int hist[], uniform int64 perm[])
{
  if (taskIndex >= taskCount) return;
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;
  uniform int strip = (end-start)/programCount;
  uniform int tail = (end-start)%programCount;
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

task void copy (uniform int span, uniform int n, uniform int64 from[], uniform int64 to[])
{
  if (taskIndex >= taskCount) return;
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;

  foreach (i = start ... end)
  {
    to[i] = from[i];
  }
}

task void pack (uniform int span, uniform int n, uniform unsigned int code[], uniform int64 pair[])
{
  if (taskIndex >= taskCount) return;
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;

  foreach (i = start ... end)
  {
    pair[i] = ((int64)i<<32)+code[i];
  }
}

task void unpack (uniform int span, uniform int n, uniform int64 pair[], uniform int unsigned code[], uniform int order[])
{
  if (taskIndex >= taskCount) return;
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;

  foreach (i = start ... end)
  {
    code[i] = pair[i];
    order[i] = pair[i]>>32;
  }
}

task void addup (uniform int h[], uniform int g[])
{
  if (taskIndex >= taskCount) return;
  uniform int * uniform u = &h[256*programCount*taskIndex];
  uniform int i, x, y = 0;

  for (i = 0; i < 256*programCount; i ++)
  {
    x = u[i];
    u[i] = y;
    y += x;
  }

  g[taskIndex] = y;
}

task void bumpup (uniform int h[], uniform int g[])
{
  if (taskIndex >= taskCount) return;
  uniform int * uniform u = &h[256*programCount*taskIndex];
  uniform int z = g[taskIndex];

  foreach (i = 0 ... 256*programCount)
  {
    u[i] += z;
  }
}

static void prefix_sum (uniform int num, uniform int h[], uniform int g[])
{
  uniform int i;

  launch[num] addup (h, g+1);
  sync;

  g[0] = 0;
  for (i = 1; i < num; i ++) 
      g[i] += g[i-1];

  launch[num] bumpup (h, g);
  sync;

}

export void sort_ispc (uniform int n, 
    uniform unsigned int code[], 
    uniform int order[], 
    uniform int ntasks,
    uniform int _hist[],
    uniform int64 _pair[],
    uniform int64 _temp[],
    uniform int _g[])
{
  uniform int num = ntasks;
  uniform int span = n / num;
#if 1 
  /* if fails, change to 0. some issues with stack size/heap inside CUDA context
   * grep for CuCtxSetLimit in  sort_cu.cpp 
   */ 
  uniform int hsize = 256*programCount*num;
  uniform int * uniform hist = uniform new uniform int [hsize];
  uniform int64 * uniform pair = uniform new uniform int64 [n];
  uniform int64 * uniform temp = uniform new uniform int64 [n];
  uniform int * uniform g = uniform new uniform int [num+1];
#define ALLOCATED
#else
   uniform int * uniform hist = _hist;
   uniform int64 * uniform pair = _pair;
   uniform int64 * uniform temp = _temp;
   uniform int * uniform g = _g;
#endif
  uniform int pass, i;

  launch[num] pack (span, n, code, pair);
  sync;

  for (pass = 0; pass < 4; pass ++)
  {
    launch[num] histogram (span, n, pair, pass, hist);
    sync;

    prefix_sum (num, hist, g);

    launch[num] permutation (span, n, pair, pass, hist, temp);
    sync;

    launch[num] copy (span, n, temp, pair);
    sync;
  }

  launch[num] unpack (span, n, pair, code, order);
  sync;

#ifdef ALLOCATED
  delete g;
  delete hist;
  delete pair;
  delete temp;
#endif
}

