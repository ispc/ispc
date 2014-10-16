/*
  Copyright (c) 2014, Evghenii Gaburov
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
/*
   Based on radixSort from  http://www.moderngpu.com
   */

#include "cuda_helpers.cuh"
#include <cassert>

#define NUMBITS 8
#define NUMDIGITS (1<<NUMBITS)

typedef long long Key;

__forceinline__ __device__ int atomic_add_global(int* ptr, int value)
{
  return atomicAdd(ptr, value);
}

static __device__ __forceinline__ int shfl_scan_add_step(int partial, int up_offset)
{
  int result;
  asm(
      "{.reg .u32 r0;"
      ".reg .pred p;"
      "shfl.up.b32 r0|p, %1, %2, 0;"
      "@p add.u32 r0, r0, %3;"
      "mov.u32 %0, r0;}"
      : "=r"(result) : "r"(partial), "r"(up_offset), "r"(partial));
  return result;
}

__forceinline__ __device__ int exclusive_scan_add(int value)
{
  int mysum = value;
#pragma unroll
  for(int i = 0; i < 5; ++i)
    mysum = shfl_scan_add_step(mysum, 1 << i);
  return mysum - value;
}

__global__
void countPass(
    const  Key keysAll[],
    Key sortedAll[],
    const  int bit,
    const  int numElements,
    int countsAll[],
    int countsGlobal[])
{
  const  int  blkIdx = taskIndex;
  const  int numBlocks = taskCount;
  const  int  blkDim = (numElements + numBlocks - 1) / numBlocks;

  const  int mask = (1 << NUMBITS) - 1;

  const  Key *  keys   =   keysAll + blkIdx*blkDim;
  Key *  sorted = sortedAll + blkIdx*blkDim;
  int *      counts = countsAll + blkIdx*NUMDIGITS;
  const  int           nloc = min(numElements - blkIdx*blkDim, blkDim);

#pragma unroll 8
  for (int digit = programIndex; digit < NUMDIGITS; digit += programCount)
    counts[digit] = 0;

  for (int i = programIndex; i < nloc; i += programCount)
    if (i < nloc)
    {
      sorted[i] = keys[i];
      const int key = mask & ((unsigned int)keys[i] >> bit);
      atomic_add_global(&counts[key], 1);
    }

#pragma unroll 8
  for (int digit = programIndex; digit < NUMDIGITS; digit += programCount)
    atomic_add_global(&countsGlobal[digit], counts[digit]);
}

__global__
void sortPass(
    Key keysAll[],
    Key sorted[],
    int bit,
    int numElements,
    int digitOffsetsAll[])
{
  const  int  blkIdx = taskIndex;
  const  int numBlocks = taskCount;

  const  int  blkDim = (numElements + numBlocks - 1) / numBlocks;


  const  int keyIndex = blkIdx * blkDim;
  Key *  keys = keysAll + keyIndex;


  const  int nloc = min(numElements - keyIndex, blkDim);

  const  int mask = (1 << NUMBITS) - 1;

  /* copy digit offset from Gmem to Lmem */
#if 1
  __shared__ int digitOffsets_sh[NUMDIGITS*4];
  volatile int *digitOffsets = digitOffsets_sh + warpIdx*NUMDIGITS;
  for (int digit = programIndex; digit < NUMDIGITS; digit += programCount)
    digitOffsets[digit] = digitOffsetsAll[blkIdx*NUMDIGITS + digit];
#else
  int *digitOffsets = &digitOffsetsAll[blkIdx*NUMDIGITS];
#endif


  for (int i = programIndex; i < nloc; i += programCount)
    if (i < nloc)
    {
      const int key = mask & ((unsigned int)keys[i] >> bit);
      int scatter;
      /* not a vector friendly loop */
#pragma unroll 1  /* needed, otherwise compiler unroll and optimizes the result :S */
      for (int iv = 0; iv < programCount; iv++)
        if (programIndex == iv)
          scatter = digitOffsets[key]++;
      sorted [scatter] = keys[i];
    }
}

__global__
void partialScanLocal(
    int numBlocks,
    int excScanAll[],
    int  countsAll[],
    int partialSumAll[])
{
  const  int  blkIdx = taskIndex;

  const  int  blkDim = (numBlocks+taskCount-1)/taskCount;
  const  int      bbeg = blkIdx * blkDim;
  const  int      bend = min(bbeg + blkDim, numBlocks);

  int (*   countsBlock)[NUMDIGITS] = ( int (*)[NUMDIGITS])countsAll;
  int (*  excScanBlock)[NUMDIGITS] = ( int (*)[NUMDIGITS])excScanAll;
  int (*    partialSum)[NUMDIGITS] = ( int (*)[NUMDIGITS])partialSumAll;

#pragma unroll 8
  for (int digit = programIndex; digit < NUMDIGITS; digit += programCount)
  {
    int prev = bbeg == 0 ? excScanBlock[0][digit] : 0;
    for ( int block = bbeg; block < bend; block++)
    {
      const int y = countsBlock[block][digit];
      excScanBlock[block][digit] = prev;
      prev += y;
    }
    partialSum[blkIdx][digit] = excScanBlock[bend-1][digit] + countsBlock[bend-1][digit];
  }
}

__global__
void partialScanGlobal(
    const  int numBlocks,
    int partialSumAll[],
    int prefixSumAll[])
{
  int (*  partialSum)[NUMDIGITS] = ( int (*)[NUMDIGITS])partialSumAll;
  int (*   prefixSum)[NUMDIGITS] = ( int (*)[NUMDIGITS]) prefixSumAll;
  const  int digit = taskIndex;
  int carry = 0;
  for (int block = programIndex;  block < numBlocks; block += programCount)
  {
    const int value = partialSum[block][digit];
    const int scan  = exclusive_scan_add(value);
    if (block < numBlocks)
      prefixSum[block][digit] = scan + carry;
    carry += __shfl(scan+value, programCount-1);
  }
}

__global__
void completeScanGlobal(
    int numBlocks,
    int excScanAll[],
    int carryValueAll[])
{
  const  int  blkIdx = taskIndex;
  const  int  blkDim = (numBlocks+taskCount-1)/taskCount;
  const  int      bbeg = blkIdx * blkDim;
  const  int      bend = min(bbeg  + blkDim, numBlocks);

  int (*  excScanBlock)[NUMDIGITS] = ( int (*)[NUMDIGITS])excScanAll;
  int (*    carryValue)[NUMDIGITS] = ( int (*)[NUMDIGITS])carryValueAll;

#pragma unroll 8
  for (int digit = programIndex; digit < NUMDIGITS; digit += programCount)
  {
    const int carry = carryValue[blkIdx][digit];
    for ( int block = bbeg; block < bend; block++)
      excScanBlock[block][digit] += carry;
  }
}

__device__ static
inline void radixExclusiveScan(
    const  int numBlocks,
    int excScanPtr[],
    int  countsPtr[],
    int partialSum[],
    int  prefixSum[])
{
  const  int scale = 8;
  launch (numBlocks/scale, 1,1, partialScanLocal)(numBlocks, excScanPtr, countsPtr, partialSum);
  sync;

  launch (NUMDIGITS,1,1,partialScanGlobal) (numBlocks/scale, partialSum, prefixSum);
  sync;

  launch (numBlocks/scale,1,1, completeScanGlobal) (numBlocks, excScanPtr, prefixSum);
  sync;
}

__device__ static  int *  memoryPool = NULL;
__device__ static  int numBlocks;
__device__ static  int nSharedCounts;
__device__ static  int nCountsGlobal;
__device__ static  int nExcScan;
__device__ static  int nCountsBlock;
__device__ static  int nPartialSum;
__device__ static  int nPrefixSum;

__device__ static  int *  sharedCounts;
__device__ static  int *  countsGlobal;
__device__ static  int *  excScan;
__device__ static  int *  counts;
__device__ static  int *  partialSum;
__device__ static  int *  prefixSum;

__device__ static  int numElementsBuf = 0;
__device__ static  Key *  bufKeys;

__global__
void radixSort_alloc___export(const  int n)
{
  assert(memoryPool == NULL);
  numBlocks     = 13*32*4;
  nSharedCounts = NUMDIGITS*numBlocks;
  nCountsGlobal = NUMDIGITS;
  nExcScan      = NUMDIGITS*numBlocks;
  nCountsBlock  = NUMDIGITS*numBlocks;
  nPartialSum   = NUMDIGITS*numBlocks;
  nPrefixSum    = NUMDIGITS*numBlocks;


  const  int nalloc =
    nSharedCounts +
    nCountsGlobal +
    nExcScan +
    nCountsBlock +
    nPartialSum +
    nPrefixSum;

  if (programIndex == 0)
    memoryPool =  new  int[nalloc];

  sharedCounts = memoryPool;
  countsGlobal = sharedCounts + nSharedCounts;
  excScan      = countsGlobal + nCountsGlobal;
  counts       = excScan      + nExcScan;
  partialSum   = counts       + nCountsBlock;
  prefixSum    = partialSum   + nPartialSum;
}

extern "C"
void radixSort_alloc(const  int n)
{
  radixSort_alloc___export<<<1,32>>>(n);
  sync;
}


__device__  static
void radixSort_freeBufKeys()
{
  if (numElementsBuf > 0)
  {
    if (programIndex == 0)
      delete bufKeys;
    numElementsBuf = 0;
  }
}

__global__ void radixSort_free___export()
{
  assert(memoryPool != NULL);
  if (programIndex == 0)
    delete memoryPool;
  memoryPool = NULL;

  radixSort_freeBufKeys();
}
extern "C"
void radixSort_free()
{
  radixSort_free___export<<<1,32>>>();
  sync;
}

__global__ void radixSort___export(
    const  int numElements,
    Key keys[],
    const  int nBits)
{
#ifdef __NVPTX__
  assert((numBlocks & 3) == 0);  /* task granularity on Kepler is 4 */
#endif

  if (numElementsBuf < numElements)
    radixSort_freeBufKeys();
  if (numElementsBuf == 0)
  {
    numElementsBuf = numElements;
    if (programIndex == 0)
      bufKeys =  new  Key[numElementsBuf];
  }

  const  int blkDim  = (numElements + numBlocks - 1) / numBlocks;

  for ( int bit = 0; bit < nBits; bit += NUMBITS)
  {
    /* initialize histogram for each digit */
    for (int digit = programIndex; digit < NUMDIGITS; digit += programCount)
      countsGlobal[digit] = 0;

    /* compute histogram for each digit */
    launch (numBlocks,1,1, countPass)(keys, bufKeys, bit, numElements, counts, countsGlobal);
    sync;

    /* exclusive scan on global histogram */
    int carry = 0;
    excScan[0] = 0;
#pragma unroll 8
    for (int digit = programIndex; digit < NUMDIGITS; digit += programCount)
    {
      const int value = countsGlobal[digit];
      const int scan  = exclusive_scan_add(value);
      excScan[digit] = scan + carry;
      carry += __shfl(scan+value, programCount-1);
    }

    /* computing offsets for each digit */
    radixExclusiveScan(numBlocks, excScan, counts, partialSum, prefixSum);

    /* sorting */
    launch (numBlocks,1,1,
      sortPass)(
          bufKeys,
          keys,
          bit,
          numElements,
          excScan);
    sync;
  }
}

extern "C"
void radixSort(
    const  int numElements,
    Key keys[],
    const  int nBits)
{
  cudaDeviceSetCacheConfig ( cudaFuncCachePreferEqual );
  radixSort___export<<<1,32>>>(numElements, keys, nBits);
  sync;
}
