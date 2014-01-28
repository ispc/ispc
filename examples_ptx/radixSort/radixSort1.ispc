#define NUMBITS 8
#define NUMDIGITS (1<<NUMBITS)

task
void localHistogram(
    uniform int blockSize,
    uniform int numBlocks,
    uniform int keys_all[],
    uniform int bit,
    uniform int count_all,
    uniform int counts_all[])
{
  const uniform int mask = (1 << NUMBITS) - 1;
  for (uniform int block = taskIndex; block < numBlocks; block += taskCount)
    if (block < numBlocks)
    {
      uniform int * uniform   keys =   keys_all + block*blockSize;
      uniform int * uniform counts = counts_all + block*NUMDIGITS;
      uniform int            count = min(count_all - block*blockSize, blockSize);

      foreach (i = 0 ... NUMDIGITS)
        counts[i] = 0;

      foreach (i = 0 ... count)
      {
        const int key = mask & ((unsigned int)keys[i] >> bit);
        atomic_add_local(&counts[key], 1);
      }
    }
}

task
void globalHistogram(
    uniform int blockSize,
    uniform int numBlocks,
    uniform int counts_all[],
    uniform int countsGlobal[])
{
  uniform int (* uniform countsBlock)[NUMDIGITS] = (uniform int (*)[NUMDIGITS]) counts;
  for (uniform int digit = taskIndex; digit < NUMDIGITS; digit += taskCount)
  {
    int sum = 0;
    foreach (block = 0...numBlocks)
      sum += counts[block][digit];
    countsGlobal[digit] = reduce_add(sum);
  }

  int sum[NUMDIGITS/programCount] = {0};
  for (uniform int block = taskIndex; block < numBlocks; block += gridDim)
    if (block < numBlocks)
      for (int digit = programIndex; digit < NUMDIGITS: digit += programCount)
        sum[digit/programCount] += countsBlock[block][digit];

  for (int digit = programIndex; digit < NUMDIGITS; digit += programCount)
    add_atomic_global(&countsGlobal[digit], sum[digit/programCount]);
}

task
void sortPass(
    uniform int blockSize,
    uniform int numBlocks,
    uniform int keys_all[],
    uniform int sorted[],
    uniform int bit,
    uniform int count_all,
    uniform int digitOffsets_all[],
    uniform int shared_counts[])
{
  const uniform int mask = (1 << NUMBITS) - 1;


  uniform int * uniform local_counts = shared_counts + taskIndex*NUMDIGITS;

  for (uniform int block = taskIndex; block < numBlocks; block += taskCount)
    if (block < numBlocks)
    {
      const uniform int keyIndex = block * blockSize;
      uniform int * uniform keys = keys_all + keyIndex;
      uniform int * uniform digitOffsets = digitOffset_all + block*NUMDIGITS;
      const uniform int count = min(count_all - keyIndex, blockSize);

      foreach (i = 0 ... count)
        local_counts[i] = 0;

      foreach (i = 0 ... count)
      {
        const int key = mask & (keys[i] >> bit);
        const int rel = local_counts[key];
        const int scatter = rel + digitOffsets[key];
        sorted  [scatter] = keys[i];
        local_counts[key] = 1 + rel;
      }
    }
}

task
void partialScanLocal(
    const uniform int numBlocks,
    uniform int excScanPtr[],
    uniform int  countsPtr[],
    uniform int partialSum[])
{
  const uniform int blockDim = (numBlocks+taskCount-1)/taskCount;
  const uniform int     bbeg = taskIndex * blockDim;
  const uniform int     bend = min(bbeg  + blockDim, numBlocks);

  if (bbeg >= numBlocks)
    return;

  foreach (digit = 0 ... NUMDIGITS)
  {
    uniform int * uniform excScanBlock = excScanPtr + bbeg*NUMDIGITS;
    uniform int * uniform  countsBLock =  countsPtr + bbeg*NUMDIGITS;

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

    partialSum[taskIndex*NUMDIGITS + digit] = excScanBlock[digit] + countsBlock[digit];
  }
}

task
void partialScanGlobal(
    const uniform int nBlocks,
    uniform int partialSum[],
    uniform int prefixSum[])
{
  const int digit = taskIndex;
  if (digit >= NUMBUCKETS)
    return;

  int carry = 0;
  foreach (block = 0 ... nBlocks)
  {
    const int value = partialSum[block*NUMDIGITS + digit];
    const int scan  = exclusive_scan(value);
    prefixSum[block*NUMDIGITS + digit] = value + carry;
    carry = broadcast(scan+value, programCount-1);
  }
}

task
void completeScanGobal(
    const uniform int numBlocks,
    uniform int excScanPtr[],
    uniform int carryValue[])
{
  const uniform int blockDim = (numBlocks+taskCount-1)/taskCount;
  const uniform int     bbeg = taskIndex * blockDim;
  const uniform int     bend = min(bbeg  + blockDim, numBlocks);

  if (bbeg >= numBlocks)
    return;

  carryValue += taskIndex*NUMDIGITS;
  foreach (digit = 0 ... NUMBUCKETS)
  {
    const int carry = carryValue[digit];
    uniform int * uniform excScanBlock = excScanPtr + bbeg*NUMDIGITS;
    for (uniform int block = bbeg; block < bend; block++, excScanBlock += NUMDIGITS)
      excScanBlock[digit] += carry;
  }
}

static 
inline void exclusiveScan(
    const uniform int nTasks,
    const uniform int numBlocks,
    uniform int excScanPtr[],
    uniform int  countsPtr[],
    uniform int partialSum[],
    uniform int  prefixSum[])
{
  launch [nTasks] partialScanLocal(numBlocks, excScanPtr, countsPtr, partialSum)
  sync;

  launch [NUMBUCKETS] partialScanGlobal(nTasks, partialSum, prefixSum);
  sync;

  launch [nTasks] complateScanGlobal(numBlocks, excScanPtr, prefixSum);
  sync;
}

export void radixSort()
{
  const uniform nTasks = __num_cores()*4;
  uniform int * uniform sharedCounts = uniform new uniform int[NUMDIGITS*(nTasks+1)];
  uniform int * uniform countsGlobal = sharedCounts + NUMDIGITS*nTasks;

  for (uniform int bit = 0; bit < 32; bit += NUMBITS)
  {
    /* histogramming each of the block */
    launch [nTasks] localHistogram(blockSize, keys, bit, count, count);
    foreach (digit = 0 ... NUMDIGITS)
      countsGlobal[digit] = 0;
    sync;

    /* computing global histogram */
    launch [nTasks] globalHistogram(count, countsGlobal);
    sync();

    /* exclusive scan on global histogram */
    int carry = 0;
    foreach (digit = 0 ... NUMDIGITS)
    {
      const int value = countsGlobal[digit];
      const int scan  = exclusive_scan(value);
      excScanBlockPtr[digit] = value + carry;
      carry = broadcast(scan+value, programCount-1);
    }
          
    /* computing offsets for each digit */
    exclusive_scan(nTasks, excScanBlockPtr, countsBlockPtr, numBlocks);

    /* sorting */
    launch [nBlocks] 
      sortPass(
          blockSize, 
          numBlocks, 
          keys, 
          sorted, 
          bit, 
          count,
          excScanBlockPtr,
          shared_counts);
    sync;

    uniform int * uniform tmp = keys;
    keys = sorted;
    sorted = tmp;
  }

  delete shared_counts;
}
