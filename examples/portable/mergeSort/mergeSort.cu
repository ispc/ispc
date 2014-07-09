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
   Based on mergeSort from CUDA SDK
   */

#include "keyType.h"
#include "cuda_helpers.cuh"
#include <cassert>

#define uniform

#define SAMPLE_STRIDE programCount

#define iDivUp(a,b) (((a) + (b) - 1)/(b))
#define getSampleCount(dividend) (iDivUp((dividend), (SAMPLE_STRIDE)))

#define W (/*sizeof(int)=*/4 * 8)

__device__ static inline
int nextPowerOfTwo(int x)
{
#if 0
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
#else
  return 1U << (W - __clz(x - 1));
#endif
}


__device__ static inline
int binarySearchInclusiveRanks(
    const int val,
    uniform int *data,
    const int L,
    int stride)
{
  if (L == 0)
    return 0;

  int pos = 0;
  for (; stride > 0; stride >>= 1)
  {
    int newPos = min(pos + stride, L);

    if (data[newPos - 1] <= val)
      pos = newPos;
  }

  return pos;
}

__device__ static inline
int binarySearchExclusiveRanks(
    const int val,
    uniform int *data,
    const int L,
    int stride)
{
  if (L == 0)
    return 0;

  int pos = 0;
  for (; stride > 0; stride >>= 1)
  {
    int newPos = min(pos + stride, L);

    if (data[newPos - 1] < val)
      pos = newPos;
  }

  return pos;
}

__device__ static inline
int binarySearchInclusive(
    const Key_t val,
    uniform Key_t *data,
    const int L,
    int stride)
{
  if (L == 0)
    return 0;

  int pos = 0;
  for (; stride > 0; stride >>= 1)
  {
    int newPos = min(pos + stride, L);

    if (data[newPos - 1] <= val)
      pos = newPos;
  }

  return pos;
}

__device__ static inline
int binarySearchExclusive(
    const Key_t val,
    uniform Key_t *data,
    const int L,
    int stride)
{
  if (L == 0)
    return 0;

  int pos = 0;
  for (; stride > 0; stride >>= 1)
  {
    int newPos = min(pos + stride, L);

    if (data[newPos - 1] < val)
      pos = newPos;
  }

  return pos;
}

__device__ static inline
int binarySearchInclusive1(
    const Key_t val,
    Key_t data,
    const uniform int L,
    uniform int stride)
{
  if (L == 0)
    return 0;

  int pos = 0;
  for (; stride > 0; stride >>= 1)
  {
    int newPos = min(pos + stride, L);

    if (shuffle(data,newPos - 1) <= val)
      pos = newPos;
  }

  return pos;
}

__device__ static inline
int binarySearchExclusive1(
    const Key_t val,
    Key_t data,
    const uniform int L,
    uniform int stride)
{
  if (L == 0)
    return 0;

  int pos = 0;
  for (; stride > 0; stride >>= 1)
  {
    int newPos = min(pos + stride, L);

    if (shuffle(data,newPos - 1) < val)
      pos = newPos;
  }

  return pos;
}

////////////////////////////////////////////////////////////////////////////////
// Bottom-level merge sort (binary search-based)
////////////////////////////////////////////////////////////////////////////////
__global__
void mergeSortGangKernel(
    uniform int batchSize,
    uniform Key_t dstKey[],
    uniform Val_t dstVal[],
    uniform Key_t srcKey[],
    uniform Val_t srcVal[])
{
  const uniform int blkIdx = taskIndex;
  const uniform int blkDim = (batchSize + taskCount - 1)/taskCount;
  const uniform int blkBeg =     blkIdx * blkDim;
  const uniform int blkEnd = min(blkBeg + blkDim, batchSize);

  __shared__ Key_t s_key_tmp[2*programCount*4];
  __shared__ Val_t s_val_tmp[2*programCount*4];
  Key_t *s_key = s_key_tmp + warpIdx*(2*programCount);
  Val_t *s_val = s_val_tmp + warpIdx*(2*programCount);

  for (uniform int blk = blkBeg; blk < blkEnd; blk++)
  {
    const uniform int base = blk * (programCount*2);
    s_key[programIndex +            0] = srcKey[base + programIndex +            0];
    s_val[programIndex +            0] = srcVal[base + programIndex +            0];
    s_key[programIndex + programCount] = srcKey[base + programIndex + programCount];
    s_val[programIndex + programCount] = srcVal[base + programIndex + programCount];

    for (uniform int stride = 1; stride < 2*programCount; stride <<= 1)
    {
      const int lPos = programIndex & (stride - 1);
      uniform Key_t *baseKey = s_key + 2 * (programIndex - lPos);
      uniform Val_t *baseVal = s_val + 2 * (programIndex - lPos);

      Key_t keyA = baseKey[lPos +      0];
      Val_t valA = baseVal[lPos +      0];
      Key_t keyB = baseKey[lPos + stride];
      Val_t valB = baseVal[lPos + stride];
      int posA = binarySearchExclusive(keyA, baseKey + stride, stride, stride) + lPos;
      int posB = binarySearchInclusive(keyB, baseKey +      0, stride, stride) + lPos;

      baseKey[posA] = keyA;
      baseVal[posA] = valA;
      baseKey[posB] = keyB;
      baseVal[posB] = valB;
    }

    dstKey[base + programIndex +            0] = s_key[programIndex +            0];
    dstVal[base + programIndex +            0] = s_val[programIndex +            0];
    dstKey[base + programIndex + programCount] = s_key[programIndex + programCount];
    dstVal[base + programIndex + programCount] = s_val[programIndex + programCount];
  }
}

__device__ static inline
void mergeSortGang(
    uniform Key_t dstKey[],
    uniform Val_t dstVal[],
    uniform Key_t srcKey[],
    uniform Val_t srcVal[],
    uniform int batchSize)
{
  uniform int nTasks = batchSize;
  launch (nTasks,1,1,mergeSortGangKernel)(batchSize, dstKey, dstVal, srcKey, srcVal);
  sync;
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 1: generate sample ranks
////////////////////////////////////////////////////////////////////////////////
__global__
void generateSampleRanksKernel(
    uniform int nBlocks,
    uniform int in_ranksA[],
    uniform int in_ranksB[],
    uniform Key_t in_srcKey[],
    uniform int stride,
    uniform int N,
    uniform int totalProgramCount)
{
  const uniform int blkIdx = taskIndex;
  const uniform int blkDim = (nBlocks + taskCount - 1)/taskCount;
  const uniform int blkBeg =     blkIdx * blkDim;
  const uniform int blkEnd = min(blkBeg + blkDim, nBlocks);

  for (uniform int blk = blkBeg; blk < blkEnd; blk++)
  {
    const int pos = blk * programCount + programIndex;
    cif (pos >= totalProgramCount)
      return;

    const int           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);

    uniform Key_t * srcKey = in_srcKey + segmentBase;
    uniform int * ranksA = in_ranksA + segmentBase / SAMPLE_STRIDE;
    uniform int * ranksB = in_ranksB + segmentBase / SAMPLE_STRIDE;

    const int segmentElementsA = stride;
    const int segmentElementsB = min(stride, N - segmentBase - stride);
    const int  segmentSamplesA = getSampleCount(segmentElementsA);
    const int  segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
      ranksA[i] = i * SAMPLE_STRIDE;
      ranksB[i] = binarySearchExclusive(
          srcKey[i * SAMPLE_STRIDE], srcKey + stride,
          segmentElementsB, nextPowerOfTwo(segmentElementsB));
    }

    if (i < segmentSamplesB)
    {
      ranksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
      ranksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive(
          srcKey[stride + i * SAMPLE_STRIDE], srcKey + 0,
          segmentElementsA, nextPowerOfTwo(segmentElementsA));
    }
  }
}

__device__ static inline
void generateSampleRanks(
    uniform int ranksA[],
    uniform int ranksB[],
    uniform Key_t srcKey[],
    uniform int stride,
    uniform int N)
{
  uniform int lastSegmentElements = N % (2 * stride);
  uniform int threadCount = (lastSegmentElements > stride) ?
    (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) :
    (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

  uniform int nBlocks = iDivUp(threadCount, SAMPLE_STRIDE);
  uniform int nTasks = nBlocks;

  launch (nTasks,1,1, generateSampleRanksKernel)(nBlocks, ranksA, ranksB, srcKey, stride, N, threadCount);
  sync;
}
////////////////////////////////////////////////////////////////////////////////
// Merge step 2: generate sample ranks and indices
////////////////////////////////////////////////////////////////////////////////
__global__
void mergeRanksAndIndicesKernel(
    uniform int nBlocks,
    uniform int in_Limits[],
    uniform int in_Ranks[],
    uniform int stride,
    uniform int N,
    uniform int totalProgramCount)
{
  const uniform int blkIdx = taskIndex;
  const uniform int blkDim = (nBlocks + taskCount - 1)/taskCount;
  const uniform int blkBeg =     blkIdx * blkDim;
  const uniform int blkEnd = min(blkBeg + blkDim, nBlocks);

  for (uniform int blk = blkBeg; blk < blkEnd; blk++)
  {
    int pos = blk * programCount + programIndex;
    cif (pos >= totalProgramCount)
      return;

    const int           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    uniform int *  ranks = in_Ranks  + (pos - i) * 2;
    uniform int * limits = in_Limits + (pos - i) * 2;

    const int segmentElementsA = stride;
    const int segmentElementsB = min(stride, N - segmentBase - stride);
    const int  segmentSamplesA = getSampleCount(segmentElementsA);
    const int  segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
      int dstPos = binarySearchExclusiveRanks(ranks[i], ranks + segmentSamplesA, segmentSamplesB, nextPowerOfTwo(segmentSamplesB)) + i;
      limits[dstPos] = ranks[i];
    }

    if (i < segmentSamplesB)
    {
      int dstPos = binarySearchInclusiveRanks(ranks[segmentSamplesA + i], ranks, segmentSamplesA, nextPowerOfTwo(segmentSamplesA)) + i;
      limits[dstPos] = ranks[segmentSamplesA + i];
    }
  }
}
__device__ static inline
void mergeRanksAndIndices(
    uniform int limitsA[],
    uniform int limitsB[],
    uniform int ranksA[],
    uniform int ranksB[],
    uniform int stride,
    uniform int N)
{
  const uniform int lastSegmentElements = N % (2 * stride);
  const uniform int threadCount = (lastSegmentElements > stride) ?
    (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) :
    (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

  const uniform int nBlocks = iDivUp(threadCount, SAMPLE_STRIDE);
  uniform int nTasks = nBlocks;

  launch (nTasks,1,1,mergeRanksAndIndicesKernel)(
      nBlocks,
      limitsA,
      ranksA,
      stride,
      N,
      threadCount);
  launch (nTasks,1,1, mergeRanksAndIndicesKernel)(
      nBlocks,
      limitsB,
      ranksB,
      stride,
      N,
      threadCount);
  sync;
}


__global__
void mergeElementaryIntervalsKernel(
    uniform int mergePairs,
    uniform Key_t dstKey[],
    uniform Val_t dstVal[],
    uniform Key_t srcKey[],
    uniform Val_t srcVal[],
    uniform int limitsA[],
    uniform int limitsB[],
    uniform int stride,
    uniform int N)
{
  const uniform int blkIdx = taskIndex;
  const uniform int blkDim = (mergePairs + taskCount - 1)/taskCount;
  const uniform int blkBeg =     blkIdx * blkDim;
  const uniform int blkEnd = min(blkBeg + blkDim, mergePairs);

  for (uniform int blk = blkBeg; blk < blkEnd; blk++)
  {
    const int uniform   intervalI =  blk & ((2 * stride) / SAMPLE_STRIDE - 1);
    const int uniform segmentBase = (blk - intervalI) * SAMPLE_STRIDE;

    //Set up threadblk-wide parameters

    const uniform int segmentElementsA = stride;
    const uniform int segmentElementsB = min(stride, N - segmentBase - stride);
    const uniform int  segmentSamplesA = getSampleCount(segmentElementsA);
    const uniform int  segmentSamplesB = getSampleCount(segmentElementsB);
    const uniform int   segmentSamples = segmentSamplesA + segmentSamplesB;

    const uniform int startSrcA = limitsA[blk];
    const uniform int startSrcB = limitsB[blk];
    const uniform int endSrcA   = (intervalI + 1 < segmentSamples) ? limitsA[blk + 1] : segmentElementsA;
    const uniform int endSrcB   = (intervalI + 1 < segmentSamples) ? limitsB[blk + 1] : segmentElementsB;
    const uniform int lenSrcA   = endSrcA - startSrcA;
    const uniform int lenSrcB   = endSrcB - startSrcB;
    const uniform int startDstA = startSrcA + startSrcB;
    const uniform int startDstB = startDstA + lenSrcA;

    //Load main input data

    Key_t keyA, keyB;
    Val_t valA, valB;
    if (programIndex < lenSrcA)
    {
      keyA = srcKey[segmentBase + startSrcA + programIndex];
      valA = srcVal[segmentBase + startSrcA + programIndex];
    }

    if (programIndex < lenSrcB)
    {
      keyB = srcKey[segmentBase + stride + startSrcB + programIndex];
      valB = srcVal[segmentBase + stride + startSrcB + programIndex];
    }

    // Compute destination addresses for merge data
    int dstPosA, dstPosB, dstA = -1, dstB = -1;
    if (any(programIndex < lenSrcA))
      dstPosA = binarySearchExclusive1(keyA, keyB, lenSrcB, SAMPLE_STRIDE) + programIndex;
    if (any(programIndex < lenSrcB))
      dstPosB = binarySearchInclusive1(keyB, keyA, lenSrcA, SAMPLE_STRIDE) + programIndex;

    if (programIndex < lenSrcA && dstPosA < lenSrcA)
      dstA = segmentBase + startDstA + dstPosA;
    dstPosA -= lenSrcA;
    if (programIndex < lenSrcA && dstPosA < lenSrcB)
      dstA = segmentBase + startDstB + dstPosA;

    if (programIndex < lenSrcB && dstPosB < lenSrcA)
      dstB = segmentBase + startDstA + dstPosB;
    dstPosB -= lenSrcA;
    if (programIndex < lenSrcB && dstPosB < lenSrcB)
      dstB = segmentBase + startDstB + dstPosB;

    // store merge data
    if (dstA >= 0)
    {
 //     int dstA = segmentBase + startSrcA + programIndex;
      dstKey[dstA] = keyA;
      dstVal[dstA] = valA;
    }
    if (dstB >= 0)
    {
//      int dstB = segmentBase + stride + startSrcB + programIndex;
      dstKey[dstB] = keyB;
      dstVal[dstB] = valB;
    }
  }

}


__device__ static inline
void mergeElementaryIntervals(
    uniform int nTasks,
    uniform Key_t dstKey[],
    uniform Val_t dstVal[],
    uniform Key_t srcKey[],
    uniform Val_t srcVal[],
    uniform int limitsA[],
    uniform int limitsB[],
    uniform int stride,
    uniform int N)
{
  const uniform int lastSegmentElements = N % (2 * stride);
  const uniform int mergePairs = (lastSegmentElements > stride) ? getSampleCount(N) : (N - lastSegmentElements) / SAMPLE_STRIDE;


  nTasks = mergePairs/(programCount);

  launch (nTasks,1,1, mergeElementaryIntervalsKernel)(
      mergePairs,
      dstKey,
      dstVal,
      srcKey,
      srcVal,
      limitsA,
      limitsB,
      stride,
      N);
  sync;
}

__device__ static uniform int * uniform memPool = NULL;
__device__ static uniform int * uniform ranksA;
__device__ static uniform int * uniform ranksB;
__device__ static uniform int * uniform limitsA;
__device__ static uniform int * uniform limitsB;
__device__ static uniform int nTasks;
__device__ static uniform int MAX_SAMPLE_COUNT = 0;

__global__
void openMergeSort___export()
{
  nTasks = 13*32*13;
  MAX_SAMPLE_COUNT = 8*32 * 131072 / programCount;
  assert(memPool == NULL);
  const uniform int nalloc = MAX_SAMPLE_COUNT * 4;
  memPool = uniform new uniform int[nalloc];
  ranksA  = memPool;
  ranksB  =  ranksA + MAX_SAMPLE_COUNT;
  limitsA =  ranksB + MAX_SAMPLE_COUNT;
  limitsB = limitsA + MAX_SAMPLE_COUNT;
}
extern "C"
void openMergeSort()
{
  openMergeSort___export<<<1,1>>>();
  sync;
}

__global__
void closeMergeSort___export()
{
  assert(memPool != NULL);
  delete memPool;
  memPool = NULL;
}
extern "C"
void closeMergeSort()
{
  closeMergeSort___export<<<1,1>>>();
  sync;
}

__global__
void mergeSort___export(
    uniform Key_t dstKey[],
    uniform Val_t dstVal[],
    uniform Key_t bufKey[],
    uniform Val_t bufVal[],
    uniform Key_t srcKey[],
    uniform Val_t srcVal[],
    uniform int N)
{
  uniform int stageCount = 0;
  for (uniform int stride = 2*programCount; stride < N; stride <<= 1, stageCount++);

  uniform Key_t * uniform iKey, * uniform oKey;
  uniform Val_t * uniform iVal, * uniform oVal;

  if (stageCount & 1)
  {
    iKey = bufKey;
    iVal = bufVal;
    oKey = dstKey;
    oVal = dstVal;
  }
  else
  {
    iKey = dstKey;
    iVal = dstVal;
    oKey = bufKey;
    oVal = bufVal;
  }



  assert(N <= SAMPLE_STRIDE * MAX_SAMPLE_COUNT);
  assert(N % (programCount*2) == 0);

  // k20m: 140 M/s
  {
    // k20m:  2367 M/s
    mergeSortGang(iKey, iVal, srcKey, srcVal, N/(2*programCount));

#if 1
    for (uniform int stride = 2*programCount; stride < N; stride <<= 1)
    {
      const uniform int lastSegmentElements = N % (2 * stride);

      // k20m: 271 M/s
      {
#if 1
        // k20m: 944 M/s
        {
          // k20m:  1396 M/s
          //Find sample ranks and prepare for limiters merge
          generateSampleRanks(ranksA, ranksB, iKey, stride, N);

          // k20m: 2379 M/s
          //Merge ranks and indices
          mergeRanksAndIndices(limitsA, limitsB, ranksA, ranksB, stride, N);
        }
#endif

        // k20m: 371 M/s
        //Merge elementary intervals
        mergeElementaryIntervals(nTasks, oKey, oVal, iKey, iVal, limitsA, limitsB, stride, N);
      }

      if (lastSegmentElements <= stride)
        for (int i = programIndex; i < lastSegmentElements; i += programCount)
          if (i < lastSegmentElements)
          {
            oKey[N-lastSegmentElements+i] = iKey[N-lastSegmentElements+i];
            oVal[N-lastSegmentElements+i] = iVal[N-lastSegmentElements+i];
          }


      {
        uniform Key_t * uniform tmpKey = iKey;
        iKey = oKey;
        oKey = tmpKey;
      }
      {
        uniform Val_t * uniform tmpVal = iVal;
        iVal = oVal;
        oVal = tmpVal;
      }
    }
#endif
  }
}
extern "C"
void mergeSort(
    uniform Key_t dstKey[],
    uniform Val_t dstVal[],
    uniform Key_t bufKey[],
    uniform Val_t bufVal[],
    uniform Key_t srcKey[],
    uniform Val_t srcVal[],
    uniform int N)
{
  mergeSort___export<<<1,32>>>(
      dstKey,
      dstVal,
      bufKey,
      bufVal,
      srcKey,
      srcVal,
      N);
  sync;
}
