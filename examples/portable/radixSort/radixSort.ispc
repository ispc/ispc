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

#define NUMBITS 8
#define NUMDIGITS (1<<NUMBITS)

typedef int64 Key;

task
void countPass(
    const uniform Key keysAll[],
    uniform Key sortedAll[],
    const uniform int bit,
    const uniform int numElements,
    uniform int countsAll[],
    uniform int countsGlobal[])
{
  const uniform int  blockIdx = taskIndex;
  const uniform int numBlocks = taskCount;
  const uniform int  blockDim = (numElements + numBlocks - 1) / numBlocks;

  const uniform int mask = (1 << NUMBITS) - 1;

  const uniform Key * uniform keys   =   keysAll + blockIdx*blockDim;
  uniform Key * uniform sorted = sortedAll + blockIdx*blockDim;
  uniform int * uniform     counts = countsAll + blockIdx*NUMDIGITS;
  const uniform int           nloc = min(numElements - blockIdx*blockDim, blockDim);

  foreach (digit = 0 ... NUMDIGITS)
    counts[digit] = 0;

  foreach (i = 0 ... nloc)
  {
    sorted[i] = keys[i];
    const int key = mask & ((unsigned int)keys[i] >> bit);
#ifdef __NVPTX__
    atomic_add_global(&counts[key], 1);
#else
    atomic_add_local(&counts[key], 1);
#endif
  }

  foreach (digit = 0 ... NUMDIGITS)
    atomic_add_global(&countsGlobal[digit], counts[digit]);
}

task
void sortPass(
    uniform Key keysAll[],
    uniform Key sorted[],
    uniform int bit,
    uniform int numElements,
    uniform int digitOffsetsAll[])
{
  const uniform int  blockIdx = taskIndex;
  const uniform int numBlocks = taskCount;

  const uniform int  blockDim = (numElements + numBlocks - 1) / numBlocks;


  const uniform int keyIndex = blockIdx * blockDim;
  uniform Key * uniform keys = keysAll + keyIndex;


  const uniform int nloc = min(numElements - keyIndex, blockDim);

  const uniform int mask = (1 << NUMBITS) - 1;

  /* copy digit offset from Gmem to Lmem */
#if 1
  uniform int digitOffsets[NUMDIGITS];
  foreach (digit = 0 ... NUMDIGITS)
    digitOffsets[digit] = digitOffsetsAll[blockIdx*NUMDIGITS + digit];
#else
  uniform int * uniform digitOffsets = &digitOffsetsAll[blockIdx*NUMDIGITS];
#endif

  foreach (i = 0 ... nloc)
  {
    const int key = mask & ((unsigned int)keys[i] >> bit);
    int scatter;
    /* not a vector friendly loop */
    foreach_active(iv)
      scatter = digitOffsets[key]++;
    sorted[scatter] = keys[i];
  }
}

task
void partialScanLocal(
    uniform int numBlocks,
    uniform int excScanAll[],
    uniform int  countsAll[],
    uniform int partialSumAll[])
{
  const uniform int  blockIdx = taskIndex;

  const uniform int  blockDim = (numBlocks+taskCount-1)/taskCount;
  const uniform int      bbeg = blockIdx * blockDim;
  const uniform int      bend = min(bbeg + blockDim, numBlocks);

  uniform int (* uniform  countsBlock)[NUMDIGITS] = (uniform int (*)[NUMDIGITS])countsAll;
  uniform int (* uniform excScanBlock)[NUMDIGITS] = (uniform int (*)[NUMDIGITS])excScanAll;
  uniform int (* uniform   partialSum)[NUMDIGITS] = (uniform int (*)[NUMDIGITS])partialSumAll;

  foreach (digit = 0 ... NUMDIGITS)
  {
    int prev = bbeg == 0 ? excScanBlock[0][digit] : 0;
    for (uniform int block = bbeg; block < bend; block++)
    {
      const int y = countsBlock[block][digit];
      excScanBlock[block][digit] = prev;
      prev += y;
    }
    partialSum[blockIdx][digit] = excScanBlock[bend-1][digit] + countsBlock[bend-1][digit];
  }
}

task
void partialScanGlobal(
    const uniform int numBlocks,
    uniform int partialSumAll[],
    uniform int prefixSumAll[])
{
  uniform int (* uniform partialSum)[NUMDIGITS] = (uniform int (*)[NUMDIGITS])partialSumAll;
  uniform int (* uniform  prefixSum)[NUMDIGITS] = (uniform int (*)[NUMDIGITS]) prefixSumAll;
  const uniform int digit = taskIndex;
  int carry = 0;
  foreach (block = 0 ... numBlocks)
  {
    const int value = partialSum[block][digit];
    const int scan  = exclusive_scan_add(value);
    prefixSum[block][digit] = scan + carry;
    carry += broadcast(scan+value, programCount-1);
  }
}

task
void completeScanGlobal(
    uniform int numBlocks,
    uniform int excScanAll[],
    uniform int carryValueAll[])
{
  const uniform int  blockIdx = taskIndex;
  const uniform int  blockDim = (numBlocks+taskCount-1)/taskCount;
  const uniform int      bbeg = blockIdx * blockDim;
  const uniform int      bend = min(bbeg  + blockDim, numBlocks);

  uniform int (* uniform excScanBlock)[NUMDIGITS] = (uniform int (*)[NUMDIGITS])excScanAll;
  uniform int (* uniform   carryValue)[NUMDIGITS] = (uniform int (*)[NUMDIGITS])carryValueAll;

  foreach (digit = 0 ... NUMDIGITS)
  {
    const int carry = carryValue[blockIdx][digit];
    for (uniform int block = bbeg; block < bend; block++)
      excScanBlock[block][digit] += carry;
  }
}

static
inline void radixExclusiveScan(
    const uniform int numBlocks,
    uniform int excScanPtr[],
    uniform int  countsPtr[],
    uniform int partialSum[],
    uniform int  prefixSum[])
{
  const uniform int scale = 8;
  launch [numBlocks/scale] partialScanLocal(numBlocks, excScanPtr, countsPtr, partialSum);
  sync;

  launch [NUMDIGITS] partialScanGlobal(numBlocks/scale, partialSum, prefixSum);
  sync;

  launch [numBlocks/scale] completeScanGlobal(numBlocks, excScanPtr, prefixSum);
  sync;
}

static uniform int * uniform memoryPool = NULL;
static uniform int numBlocks;
static uniform int nSharedCounts;
static uniform int nCountsGlobal;
static uniform int nExcScan;
static uniform int nCountsBlock;
static uniform int nPartialSum;
static uniform int nPrefixSum;

static uniform int * uniform sharedCounts;
static uniform int * uniform countsGlobal;
static uniform int * uniform excScan;
static uniform int * uniform counts;
static uniform int * uniform partialSum;
static uniform int * uniform prefixSum;

static uniform int numElementsBuf = 0;
static uniform Key * uniform bufKeys;

export void radixSort_alloc(const uniform int n)
{
  assert(memoryPool == NULL);
  numBlocks     = num_cores()*4;
#ifdef __NVPTX__
  numBlocks     = 13*32*4; //num_cores()*4;
#endif
  nSharedCounts = NUMDIGITS*numBlocks;
  nCountsGlobal = NUMDIGITS;
  nExcScan      = NUMDIGITS*numBlocks;
  nCountsBlock  = NUMDIGITS*numBlocks;
  nPartialSum   = NUMDIGITS*numBlocks;
  nPrefixSum    = NUMDIGITS*numBlocks;


  const uniform int nalloc =
    nSharedCounts +
    nCountsGlobal +
    nExcScan +
    nCountsBlock +
    nPartialSum +
    nPrefixSum;

  memoryPool = uniform new uniform int[nalloc];

  sharedCounts = memoryPool;
  countsGlobal = sharedCounts + nSharedCounts;
  excScan      = countsGlobal + nCountsGlobal;
  counts       = excScan      + nExcScan;
  partialSum   = counts       + nCountsBlock;
  prefixSum    = partialSum   + nPartialSum;
}

static
void radixSort_freeBufKeys()
{
  if (numElementsBuf > 0)
  {
    delete bufKeys;
    numElementsBuf = 0;
  }
}

export void radixSort_free()
{
  assert(memoryPool != NULL);
  delete memoryPool;
  memoryPool = NULL;

  radixSort_freeBufKeys();
}

export void radixSort(
    const uniform int numElements,
    uniform Key keys[],
    const uniform int nBits)
{
#ifdef __NVPTX__
  assert((numBlocks & 3) == 0);  /* task granularity on Kepler is 4 */
#endif

  if (numElementsBuf < numElements)
    radixSort_freeBufKeys();
  if (numElementsBuf == 0)
  {
    numElementsBuf = numElements;
    bufKeys = uniform new uniform Key[numElementsBuf];
  }

  const uniform int blockDim  = (numElements + numBlocks - 1) / numBlocks;

  for (uniform int bit = 0; bit < nBits; bit += NUMBITS)
  {
    /* initialize histogram for each digit */
    foreach (digit = 0 ... NUMDIGITS)
      countsGlobal[digit] = 0;

    /* compute histogram for each digit */
    launch [numBlocks] countPass(keys, bufKeys, bit, numElements, counts, countsGlobal);
    sync;

    /* exclusive scan on global histogram */
    int carry = 0;
    excScan[0] = 0;
    foreach (digit = 0 ... NUMDIGITS)
    {
      const int value = countsGlobal[digit];
      const int scan  = exclusive_scan_add(value);
      excScan[digit] = scan + carry;
      carry += broadcast(scan+value, programCount-1);
    }

    /* computing offsets for each digit */
    radixExclusiveScan(numBlocks, excScan, counts, partialSum, prefixSum);

    /* sorting */
    launch [numBlocks]
      sortPass(
          bufKeys,
          keys,
          bit,
          numElements,
          excScan);
    sync;
  }

}
