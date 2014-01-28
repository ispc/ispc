#define NUMBITS 8
#define NUMDIGITS (1<<NUMBITS)

task
void countPass(
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

#if 1
  foreach (i = 0 ... nloc)
  {
    const int key = mask & ((unsigned int)keys[i] >> bit);
    uniform int skey;
    if (reduce_equal(key, &skey) == true)
      counts[skey] += reduce_add(1);
    else
    {
#ifdef __NVPTX__
      atomic_add_global(&counts[key], 1);
#else
      atomic_add_local(&counts[key], 1);
#endif
    }
  }
#else
#endif

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


  uniform int * uniform localCounts = sharedCounts + blockIdx*NUMDIGITS;

  const uniform int keyIndex = blockIdx * blockDim;
  uniform int * uniform keys = keysAll + keyIndex;
  uniform int * uniform digitOffsets = digitOffsetsAll + blockIdx*NUMDIGITS;
  const uniform int nloc = min(numElements - keyIndex, blockDim);

  const uniform int mask = (1 << NUMBITS) - 1;
  foreach (i = 0 ... NUMDIGITS)
    localCounts[i] = 0;

  foreach (i = 0 ... nloc)
  {
    const int key = mask & ((unsigned int)keys[i] >> bit);
    int rel;
    foreach_active(iv)
    {
      rel = localCounts[key];
      localCounts[key]++;
    }
    const int scatter = rel + digitOffsets[key];
    sorted [scatter] = keys[i];
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
  const uniform int scale = 2;
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
static uniform int * uniform bufKeys;

export void radixSort_alloc(const uniform int n)
{
  assert(memoryPool == NULL);
  numBlocks     = num_cores()*4;
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

  radixSort_freeBufKeys;
}

export void radixSort(
    const uniform int numElements,
    uniform int keys[])
{
#ifdef __NVPTX__
  assert((numBlocks & 3) == 0);  /* task granularity on Kepler is 4 */
#endif

  if (numElementsBuf < numElements)
    radixSort_freeBufKeys();
  if (numElementsBuf == 0)
  {
    numElementsBuf = numElements;
    bufKeys = uniform new uniform int[numElementsBuf];
  }

  const uniform int blockDim  = (numElements + numBlocks - 1) / numBlocks;

  for (uniform int bit = 0; bit < 32; bit += NUMBITS)
  {
    /* initialize histogram for each digit */
    foreach (digit = 0 ... NUMDIGITS)
      countsGlobal[digit] = 0;

    /* compute histogram for each digit */
    launch [numBlocks] countPass(keys, bit, numElements, counts, countsGlobal);
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
          keys, 
          bufKeys, 
          bit, 
          numElements,
          excScan,
          sharedCounts);
    sync;

    uniform int * uniform tmp = keys;
    keys = bufKeys;
    bufKeys = tmp;
  }

}
