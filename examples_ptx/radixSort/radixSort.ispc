#define NUMBITS 8
#define NUMDIGITS (1<<NUMBITS)

task
void computeHistogram(
    const uniform int keysAll[],
    const uniform int bit,
    const uniform int numElements,
    uniform int countsAll[],
    uniform int countsGlobal[])
{
  const uniform int  blockIdx = taskIndex;
  const uniform int numBlocks = taskCount;
  const uniform int  blockDim = (numElements + numBlocks - 1) / numBlocks;

  const uniform int mask = (1 << NUMBITS) - 1;

  const uniform int * uniform keys =   keysAll + blockIdx*blockDim;
  uniform int * uniform     counts = countsAll + blockIdx*NUMDIGITS;
  const uniform int           nloc = min(numElements - blockIdx*blockDim, blockDim);

  foreach (digit = 0 ... NUMDIGITS)
    counts[digit] = 0;

  foreach (i = 0 ... nloc)
  {
    const int key = mask & ((unsigned int)keys[i] >> bit);
    atomic_add_local(&counts[key], 1);
  }

  foreach (digit = 0 ... NUMDIGITS)
    atomic_add_global(&countsGlobal[digit], counts[digit]);
}

task
void sortPass(
    uniform int keysAll[],
    uniform int sorted[],
    uniform int bit,
    uniform int numElements,
    uniform int digitOffsetsAll[],
    uniform int sharedCounts[])
{
  const uniform int  blockIdx = taskIndex;
  const uniform int numBlocks = taskCount;

  const uniform int  blockDim = (numElements + numBlocks - 1) / numBlocks;

  const uniform int mask = (1 << NUMBITS) - 1;

  uniform int * uniform localCounts = sharedCounts + blockIdx*NUMDIGITS;

  const uniform int keyIndex = blockIdx * blockDim;
  uniform int * uniform keys = keysAll + keyIndex;
  uniform int * uniform digitOffsets = digitOffsetsAll + blockIdx*NUMDIGITS;
  const uniform int nloc = min(numElements - keyIndex, blockDim);

  foreach (i = 0 ... NUMDIGITS)
    localCounts[i] = 0;

  foreach (i = 0 ... nloc)
  {
    const int key = mask & ((unsigned int)keys[i] >> bit);
    const int rel = localCounts[key];
    const int scatter = rel + digitOffsets[key];
    sorted [scatter]  = keys[i];
    localCounts[key]  = 1 + rel;
  }
}

task
void partialScanLocal(
    uniform int excScanPtr[],
    uniform int  countsPtr[],
    uniform int partialSum[])
{
  const uniform int numBlocks = taskCount;
  const uniform int  blockIdx = taskIndex;

  const uniform int  blockDim = (numBlocks+taskCount-1)/taskCount;
  const uniform int      bbeg = blockIdx * blockDim;
  const uniform int      bend = min(bbeg + blockDim, numBlocks);

  foreach (digit = 0 ... NUMDIGITS)
  {
    uniform int * uniform excScanBlock = excScanPtr + bbeg*NUMDIGITS;
    uniform int * uniform  countsBlock =  countsPtr + bbeg*NUMDIGITS;

    int prev = bbeg == 0 ? excScanBlock[digit] : 0;
    for (uniform int block = bbeg; block < bend; block++)
    {
      const int y = countsBlock[digit];
      excScanBlock[digit] = prev;
      prev += y;

      excScanBlock += NUMDIGITS;
      countsBlock  += NUMDIGITS;
    }

    excScanBlock -= NUMDIGITS;
    countsBlock  -= NUMDIGITS;

    partialSum[blockIdx*NUMDIGITS + digit] = excScanBlock[digit] + countsBlock[digit];
  }
}

task
void partialScanGlobal(
    const uniform int numBlocks,
    uniform int partialSum[],
    uniform int prefixSum[])
{
  const int digit = taskIndex;
  int carry = 0;
  foreach (block = 0 ... numBlocks)
  {
    const int value = partialSum[block*NUMDIGITS + digit];
    const int scan  = exclusive_scan_add(value);
    prefixSum[block*NUMDIGITS + digit] = value + carry;
    carry = broadcast(scan+value, programCount-1);
  }
}

task
void completeScanGlobal(
    uniform int excScanAll[],
    uniform int carryValue[])
{
  const uniform int numBlocks = taskCount;
  const uniform int  blockIdx = taskIndex;
  const uniform int  blockDim = (numBlocks+taskCount-1)/taskCount;
  const uniform int      bbeg = blockIdx * blockDim;
  const uniform int      bend = min(bbeg  + blockDim, numBlocks);

  carryValue += blockIdx*NUMDIGITS;
  foreach (digit = 0 ... NUMDIGITS)
  {
    const int carry = carryValue[digit];
    uniform int * uniform excScanBlock = excScanAll + bbeg*NUMDIGITS;
    for (uniform int block = bbeg; block < bend; block++, excScanBlock += NUMDIGITS)
      excScanBlock[digit] += carry;
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
  launch [numBlocks] partialScanLocal(excScanPtr, countsPtr, partialSum);
  sync;

  launch [NUMDIGITS] partialScanGlobal(numBlocks, partialSum, prefixSum);
  sync;

  launch [numBlocks] completeScanGlobal(excScanPtr, prefixSum);
  sync;
}

export void radixSort(
    const uniform int numElements,
    uniform int keys[],
    uniform int sorted[])
{
  const uniform int numBlocks =  num_cores()*2;

#ifdef __NVPTX__
  assert((numBlocks & 3) == 0);  /* task granularity on Kepler is 4 */
#endif

  const uniform int blockDim  = (numElements + numBlocks - 1) / numBlocks;

  const uniform int nSharedCounts = NUMDIGITS*numBlocks;
  const uniform int nCountsGlobal = NUMDIGITS;
  const uniform int nExcScan      = NUMDIGITS*numBlocks;
  const uniform int nCountsBlock  = NUMDIGITS*numBlocks;
  const uniform int nPartialSum   = NUMDIGITS*numBlocks;
  const uniform int nPrefixSum    = NUMDIGITS*numBlocks;

  const uniform int nalloc = 
    nSharedCounts +
    nCountsGlobal +
    nExcScan +
    nCountsBlock + 
    nPartialSum +
    nPrefixSum;

  uniform int * uniform   mem_pool   = uniform new uniform int[nalloc];

  uniform int * uniform sharedCounts = mem_pool;
  uniform int * uniform countsGlobal = sharedCounts + nSharedCounts;
  uniform int * uniform excScan      = countsGlobal + nCountsGlobal;
  uniform int * uniform countsBlock  = excScan      + nExcScan;
  uniform int * uniform partialSum   = countsBlock  + nCountsBlock;
  uniform int * uniform prefixSum    = partialSum   + nPartialSum;

  for (uniform int bit = 0; bit < 32; bit += NUMBITS)
  {
    /* initialize histogram for each digit */
    foreach (digit = 0 ... NUMDIGITS)
      countsGlobal[digit] = 0;

    /* compute histogram for each digit */
    launch [numBlocks] computeHistogram(keys, bit, numElements, countsBlock, countsGlobal);
    sync;

    /* exclusive scan on global histogram */
    int carry = 0;
    excScan[0] = 0;
    foreach (digit = 0 ... NUMDIGITS)
    {
      const int value = countsGlobal[digit];
      const int scan  = exclusive_scan_add(value);
      excScan[digit] = value + carry;
      carry += broadcast(scan+value, programCount-1);
    }

          
    /* computing offsets for each digit */
    radixExclusiveScan(numBlocks, excScan, countsBlock, partialSum, prefixSum);

    /* sorting */
    launch [numBlocks] 
      sortPass(
          keys, 
          sorted, 
          bit, 
          numElements,
          excScan,
          sharedCounts);
    sync;

    uniform int * uniform tmp = keys;
    keys = sorted;
    sorted = tmp;
  }

  delete mem_pool;
}
