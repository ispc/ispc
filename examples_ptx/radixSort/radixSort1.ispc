#define NUMBITS 8
#define NUMBUCKETS (1<<NUMBITS)
task
void localHistogram(
    uniform unsigned int32 keys_all[],
    uniform          int32 bit,
    uniform          int32 count_all,
    uniform          int32 counts_all[])
{
  const uniform unsigned int mask = (1 << NUMBITS) - 1;
  for (uniform int block = taskIndex; block < numBlocks; block += taskCount)
  {
    uniform unsigned int32 * uniform keys =   keys_all + block*blockSize;
    uniform          int32 * uniform keys = counts_all + block*NUMBUCKETS;
    uniform          int32 count = min(count_all - block*blockSize, blockSize);

    foreach (i = 0 ... NUMBUCKETS)
      counts[i] = 0;

    foreach (i = 0 ... count)
    {
      const int key = mask & (keys[i] >> bit);
      atomic_add_local(&counts[key], 1);
    }
  }
}

task
void globalHistogram(
    uniform int32 counts_all[],
    uniform int32 countsGlobal[])
{
  uniform int32 (* uniform countsBlock)[NUMBUCKETS] = (uniform int (*)[NUMBUCKETS]) counts;
  for (uniform int digit = taskIndex; digit < NUMBUCKETS; digit += taskCount)
  {
    int sum = 0;
    foreach (block = 0...numBlocks)
      sum += counts[block][digit];
    countsGlobal[digit] = reduce_add(sum);
  }
}

export void radixSort()
{
  for (uniform int bit = 0; bit < 32; bit += NUMBITS)
  {
    /* histogramming each of the block */
    launch [nBlocks] localHistogram(keys, bit, count, counts);
    sync;

    /* compute global histogram */
    launch [nBlocks] globalHistogram(counts, countsGlobal);
    sync();

    /* exclusive scan on global histogram */
    int carry = 0;
    foreach (i = 0...NUMBUCKETS)
    {
      const int value = countsGlobal[i];
      const int scan  = exclusive_scan(value);
      scanGlobal[i] = value + carry;
      carry = broadcast(scan+value, programCount-1);
    }
          
    /* computing offsets for each digit */
    launch [nBlocks] computeGlobalOffset();
    sync();

    /* sorting */
    launch [nBlocks] sort()

  }
}
