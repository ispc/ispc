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

#define SAMPLE_STRIDE programCount

#define iDivUp(a,b) (((a) + (b) - 1)/(b))
#define getSampleCount(dividend) (iDivUp((dividend), (SAMPLE_STRIDE)))

#define W (/*sizeof(int)=*/4 * 8)

static inline
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
  return 1U << (W - count_leading_zeros(x - 1));
#endif
}

static inline
int binarySearchInclusiveRanks(
    const int val,
    uniform int *data,
    const int L,
    int stride)
{
  cif (L == 0)
    return 0;

  int pos = 0;
  cfor (; stride > 0; stride >>= 1)
  {
    int newPos = min(pos + stride, L);

    cif (data[newPos - 1] <= val)
      pos = newPos;
  }

  return pos;
}

static inline
int binarySearchExclusiveRanks(
    const int val,
    uniform int *data,
    const int L,
    int stride)
{
  cif (L == 0)
    return 0;

  int pos = 0;
  cfor (; stride > 0; stride >>= 1)
  {
    int newPos = min(pos + stride, L);

    if (data[newPos - 1] < val)
      pos = newPos;
  }

  return pos;
}

static inline
int binarySearchInclusive(
    const Key_t val,
    uniform Key_t *data,
    const int L,
    int stride)
{
  cif (L == 0)
    return 0;

  int pos = 0;
  cfor (; stride > 0; stride >>= 1)
  {
    int newPos = min(pos + stride, L);

    if (data[newPos - 1] <= val)
      pos = newPos;
  }

  return pos;
}

static inline
int binarySearchExclusive(
    const Key_t val,
    uniform Key_t *data,
    const int L,
    int stride)
{
  cif (L == 0)
    return 0;

  int pos = 0;
  cfor (; stride > 0; stride >>= 1)
  {
    int newPos = min(pos + stride, L);

    if (data[newPos - 1] < val)
      pos = newPos;
  }

  return pos;
}

static inline
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

static inline
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
task
void mergeSortGangKernel(
    uniform int batchSize,
    uniform Key_t dstKey[],
    uniform Val_t dstVal[],
    uniform Key_t srcKey[],
    uniform Val_t srcVal[],
    uniform int arrayLength)
{
  const uniform int blockIdx = taskIndex;
  const uniform int blockDim = (batchSize + taskCount - 1)/taskCount;
  const uniform int blockBeg =     blockIdx * blockDim;
  const uniform int blockEnd = min(blockBeg + blockDim, batchSize);

  uniform Key_t s_key[2*programCount];
  uniform Val_t s_val[2*programCount];

  for (uniform int block = blockBeg; block < blockEnd; block++)
  {
    const uniform int base = block * (programCount*2);
    s_key[programIndex +            0] = srcKey[base + programIndex +            0];
    s_val[programIndex +            0] = srcVal[base + programIndex +            0];
    s_key[programIndex + programCount] = srcKey[base + programIndex + programCount];
    s_val[programIndex + programCount] = srcVal[base + programIndex + programCount];

    for (uniform int stride = 1; stride < arrayLength; stride <<= 1)
    {
      const int lPos = programIndex & (stride - 1);
      const int offset = 2 * (programIndex - lPos);
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

static inline
void mergeSortGang(
    uniform Key_t dstKey[],
    uniform Val_t dstVal[],
    uniform Key_t srcKey[],
    uniform Val_t srcVal[],
    uniform int batchSize)
{
  uniform int nTasks = num_cores()*4;
#ifdef __NVPTX__
  nTasks = iDivUp(batchSize,1);
#endif
  launch [nTasks] mergeSortGangKernel(batchSize, dstKey, dstVal, srcKey, srcVal, 2*programCount);
  sync;
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 1: generate sample ranks
////////////////////////////////////////////////////////////////////////////////
task
void generateSampleRanksKernel(
    uniform int nBlocks,
    uniform int in_ranksA[],
    uniform int in_ranksB[],
    uniform Key_t in_srcKey[],
    uniform int stride,
    uniform int N,
    uniform int totalProgramCount)
{
  const uniform int blockIdx = taskIndex;
  const uniform int blockDim = (nBlocks + taskCount - 1)/taskCount;
  const uniform int blockBeg =     blockIdx * blockDim;
  const uniform int blockEnd = min(blockBeg + blockDim, nBlocks);

  for (uniform int block = blockBeg; block < blockEnd; block++)
  {
    const int pos = block * programCount + programIndex;
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

static inline
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
  uniform int nTasks = num_cores()*4;
#ifdef __NVPTX__
  nTasks = iDivUp(nBlocks,1);
#endif

  launch [nTasks] generateSampleRanksKernel(nBlocks, ranksA, ranksB, srcKey, stride, N, threadCount);
  sync;
}
////////////////////////////////////////////////////////////////////////////////
// Merge step 2: generate sample ranks and indices
////////////////////////////////////////////////////////////////////////////////
task
void mergeRanksAndIndicesKernel(
    uniform int nBlocks,
    uniform int in_Limits[],
    uniform int in_Ranks[],
    uniform int stride,
    uniform int N,
    uniform int totalProgramCount)
{
  const uniform int blockIdx = taskIndex;
  const uniform int blockDim = (nBlocks + taskCount - 1)/taskCount;
  const uniform int blockBeg =     blockIdx * blockDim;
  const uniform int blockEnd = min(blockBeg + blockDim, nBlocks);

  for (uniform int block = blockBeg; block < blockEnd; block++)
  {
    int pos = block * programCount + programIndex;
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
static inline
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
  uniform int nTasks = num_cores()*4;

#ifdef __NVPTX__
  nTasks = iDivUp(nBlocks,1);
#endif

  launch [nTasks] mergeRanksAndIndicesKernel(
      nBlocks,
      limitsA,
      ranksA,
      stride,
      N,
      threadCount);
  launch [nTasks] mergeRanksAndIndicesKernel(
      nBlocks,
      limitsB,
      ranksB,
      stride,
      N,
      threadCount);
  sync;
}


task
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
  const uniform int blockIdx = taskIndex;
  const uniform int blockDim = (mergePairs + taskCount - 1)/taskCount;
  const uniform int blockBeg =     blockIdx * blockDim;
  const uniform int blockEnd = min(blockBeg + blockDim, mergePairs);

  for (uniform int block = blockBeg; block < blockEnd; block++)
  {
    const int uniform   intervalI =  block & ((2 * stride) / SAMPLE_STRIDE - 1);
    const int uniform segmentBase = (block - intervalI) * SAMPLE_STRIDE;

    //Set up threadblock-wide parameters

    const uniform int segmentElementsA = stride;
    const uniform int segmentElementsB = min(stride, N - segmentBase - stride);
    const uniform int  segmentSamplesA = getSampleCount(segmentElementsA);
    const uniform int  segmentSamplesB = getSampleCount(segmentElementsB);
    const uniform int   segmentSamples = segmentSamplesA + segmentSamplesB;

    const uniform int startSrcA = limitsA[block];
    const uniform int startSrcB = limitsB[block];
    const uniform int endSrcA   = (intervalI + 1 < segmentSamples) ? limitsA[block + 1] : segmentElementsA;
    const uniform int endSrcB   = (intervalI + 1 < segmentSamples) ? limitsB[block + 1] : segmentElementsB;
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
    if (programIndex < lenSrcA)
      dstPosA = binarySearchExclusive1(keyA, keyB, lenSrcB, SAMPLE_STRIDE) + programIndex;
    if (programIndex < lenSrcB)
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

    if (dstA >= 0)
    {
      dstKey[dstA] = keyA;
      dstVal[dstA] = valA;
    }
    if (dstB >= 0)
    {
      dstKey[dstB] = keyB;
      dstVal[dstB] = valB;
    }
  }
}

static inline
void mergeElementaryIntervals(
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


  uniform int nTasks = num_cores()*4;
#ifdef __NVPTX__
  nTasks = iDivUp(mergePairs,1*programCount);
#endif

  launch [nTasks] mergeElementaryIntervalsKernel(
      mergePairs,
      dstKey,
      dstVal,
      srcKey,
      srcVal,
      limitsA,
      limitsB,
      stride,
      N);
  if (lastSegmentElements <= stride)
    foreach (i = 0 ... lastSegmentElements)
    {
      dstKey[N-lastSegmentElements+i] = srcKey[N-lastSegmentElements+i];
      dstVal[N-lastSegmentElements+i] = srcVal[N-lastSegmentElements+i];
    }
  sync;
}

static uniform int * uniform memPool = NULL;
static uniform int * uniform ranksA;
static uniform int * uniform ranksB;
static uniform int * uniform limitsA;
static uniform int * uniform limitsB;
static uniform int MAX_SAMPLE_COUNT = 0;

export
void openMergeSort()
{
  MAX_SAMPLE_COUNT = 8*32 * 131072 / programCount;
  assert(memPool == NULL);
  const uniform int nalloc = MAX_SAMPLE_COUNT * 4;
  memPool = uniform new uniform int[nalloc];
  ranksA  = memPool;
  ranksB  =  ranksA + MAX_SAMPLE_COUNT;
  limitsA =  ranksB + MAX_SAMPLE_COUNT;
  limitsB = limitsA + MAX_SAMPLE_COUNT;
}

export
void closeMergeSort()
{
  assert(memPool != NULL);
  delete memPool;
  memPool = NULL;
}

export
void mergeSort(
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

  // cpu: 28  gpu: 74 M/s
  {
    // cpu: 356   gpu: 534 M/s
    mergeSortGang(iKey, iVal, srcKey, srcVal, N/(2*programCount));

#if 1
    for (uniform int stride = 2*programCount; stride < N; stride <<= 1)
    {
      // cpu: 30  gpu: 112 M/s
      {
#if 1
        // cpu: 121  gpu: 460 M/s
        {
          // cpu: 190  gpu: 600 M/s
          //Find sample ranks and prepare for limiters merge
          generateSampleRanks(ranksA, ranksB, iKey, stride, N);

          // cpu: 120 gpu: 457 M/s
          //Merge ranks and indices
          mergeRanksAndIndices(limitsA, limitsB, ranksA, ranksB, stride, N);
        }
#endif

        // cpu: 287  gpu: 194 M/s
        //Merge elementary intervals
        mergeElementaryIntervals(oKey, oVal, iKey, iVal, limitsA, limitsB, stride, N);
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
