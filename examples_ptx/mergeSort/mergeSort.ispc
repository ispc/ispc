#define SAMPLE_STRIDE programCount

#define iDivUp(a,b) ((a) + (b) - 1)/(b)
#define getSampleCount(dividend) iDivUp((dividend), SAMPLE_STRIDE)

#define W (/*sizeof(int)=*/4 * 8)

static inline
int nextPowerOfTwo(int x)
{
#if 1
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
int binarySearchInclusive(
    const int val, 
    int *data,
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

static inline
int binarySearchExclusive(
    const int val, 
    int *data, 
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

////////////////////////////////////////////////////////////////////////////////
// Bottom-level merge sort (binary search-based)
////////////////////////////////////////////////////////////////////////////////
task 
void mergeSortGangKernel(
    uniform int dstKey[],
    uniform int dstVal[],
    uniform int srcKey[],
    uniform int srcVal[])
{
  uniform int s_key[2*programCount];
  uniform int s_val[2*programCount];

  const uniform int base = taskIndex * (programCount*2);
  s_key[programIndex +            0] = srcKey[base + programIndex +            0];
  s_val[programIndex +            0] = srcVal[base + programIndex +            0];
  s_key[programIndex + programCount] = srcKey[base + programIndex + programCount];
  s_val[programIndex + programCount] = srcVal[base + programIndex + programCount];

  for (uniform int stride = 1; stride < programCount; stride <<= 1)
  {
    const int lPos = programIndex & (stride - 1);
    int *baseKey = s_key + 2 * (programIndex - lPos);
    int *baseVal = s_val + 2 * (programIndex - lPos);

    int keyA = baseKey[lPos +      0];
    int valA = baseVal[lPos +      0];
    int keyB = baseKey[lPos + stride];
    int valB = baseVal[lPos + stride];
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

static inline
void mergeSortGang(
    uniform int dstKey[],
    uniform int dstVal[],
    uniform int srcKey[],
    uniform int srcVal[],
    uniform int batchSize)
{
  launch [batchSize] mergeSortGangKernel(dstKey, dstVal, srcKey, srcVal);
  sync;
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 1: generate sample ranks
////////////////////////////////////////////////////////////////////////////////
task
void generateSampleRanksKernel(
    uniform int in_ranksA[],
    uniform int in_ranksB[],
    uniform int in_srcKey[],
    uniform int stride,
    uniform int N,
    uniform int totalProgramCount)
{
  const int pos = taskIndex * programCount + programIndex;
  assert(pos < totalProgramCount);

  const int           i = pos & ((stride / SAMPLE_STRIDE) - 1);
  const int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);

  int * srcKey = in_srcKey + segmentBase;
  int * ranksA = in_ranksA + segmentBase / SAMPLE_STRIDE;
  int * ranksB = in_ranksB + segmentBase / SAMPLE_STRIDE;

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

static inline
void generateSampleRanks(
    uniform int ranksA[],
    uniform int ranksB[],
    uniform int srcKey[],
    uniform int stride,
    uniform int N)
{
  uniform int lastSegmentElements = N % (2 * stride);
  assert(lastSegmentElements == 0);
  uniform int threadCount = (lastSegmentElements > stride) ? 
    (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : 
    (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

  uniform int nTasks = (threadCount + programCount - 1) / programCount;

  launch [nTasks] generateSampleRanksKernel(ranksA, ranksB, srcKey, stride, N, threadCount);
  sync;
}
////////////////////////////////////////////////////////////////////////////////
// Merge step 2: generate sample ranks and indices
////////////////////////////////////////////////////////////////////////////////
task 
void mergeRanksAndIndicesKernel(
    uniform int in_Limits[],
    uniform int in_Ranks[],
    uniform int stride,
    uniform int N,
    uniform int totalProgramCount)
{
  int pos = taskIndex * programCount + programIndex;
  assert(pos < totalProgramCount);

  const int           i = pos & ((stride / SAMPLE_STRIDE) - 1);
  const int segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
  int *  ranks = in_Ranks  + (pos - i) * 2;
  int * limits = in_Limits + (pos - i) * 2;

  const int segmentElementsA = stride;
  const int segmentElementsB = min(stride, N - segmentBase - stride);
  const int  segmentSamplesA = getSampleCount(segmentElementsA);
  const int  segmentSamplesB = getSampleCount(segmentElementsB);

  if (i < segmentSamplesA)
  {
    int dstPos = binarySearchExclusive(ranks[i], ranks + segmentSamplesA, segmentSamplesB, nextPowerOfTwo(segmentSamplesB)) + i;
    limits[dstPos] = ranks[i];
  }

  if (i < segmentSamplesB)
  {
    int dstPos = binarySearchInclusive(ranks[segmentSamplesA + i], ranks, segmentSamplesA, nextPowerOfTwo(segmentSamplesA)) + i;
    limits[dstPos] = ranks[segmentSamplesA + i];
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
  assert(lastSegmentElements == 0);
  const uniform int threadCount = (lastSegmentElements > stride) ? 
    (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : 
    (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

  const uniform int nTasks = (threadCount + programCount - 1 ) / programCount;

  launch [nTasks] mergeRanksAndIndicesKernel(
      limitsA,
      ranksA,
      stride,
      N,
      threadCount);
  sync;

  launch [nTasks] mergeRanksAndIndicesKernel(
      limitsB,
      ranksB,
      stride,
      N,
      threadCount);
  sync;
}

static inline
void merge(
    uniform int dstKey[],
    uniform int dstVal[],
    uniform int srcAKey[],
    uniform int srcAVal[],
    uniform int srcBKey[],
    uniform int srcBVal[],
    uniform int lenA,
    uniform int nPowTwoLenA,
    uniform int lenB,
    uniform int nPowTwoLenB)
{
  int keyA, valA, keyB, valB, dstPosA, dstPosB;

  if (programIndex < lenA)
  {
    keyA = srcAKey[programIndex];
    valA = srcAVal[programIndex];
    dstPosA = binarySearchExclusive(keyA, srcBKey, lenB, nPowTwoLenB) + programIndex;
  }

  if (programIndex < lenB)
  {
    keyB = srcBKey[programIndex];
    valB = srcBVal[programIndex];
    dstPosB = binarySearchInclusive(keyB, srcAKey, lenA, nPowTwoLenA) + programIndex;
  }

  if (programIndex < lenA)
  {
    dstKey[dstPosA] = keyA;
    dstVal[dstPosA] = valA;
  }

  if (programIndex < lenB)
  {
    dstKey[dstPosB] = keyB;
    dstVal[dstPosB] = valB;
  }
}

task
void mergeElementaryIntervalsKernel(
    uniform int dstKey[],
    uniform int dstVal[],
    uniform int srcKey[],
    uniform int srcVal[],
    uniform int limitsA[],
    uniform int limitsB[],
    uniform int stride,
    uniform int N)
{
  uniform int s_key[2 * SAMPLE_STRIDE];
  uniform int s_val[2 * SAMPLE_STRIDE];

  const int uniform   intervalI =  taskIndex & ((2 * stride) / SAMPLE_STRIDE - 1);
  const int uniform segmentBase = (taskIndex - intervalI) * SAMPLE_STRIDE;
  srcKey += segmentBase;
  srcVal += segmentBase;
  dstKey += segmentBase;
  dstVal += segmentBase;

  //Set up threadblock-wide parameters
  uniform int startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

  {
    uniform int segmentElementsA = stride;
    uniform int segmentElementsB = min(stride, N - segmentBase - stride);
    uniform int  segmentSamplesA = getSampleCount(segmentElementsA);
    uniform int  segmentSamplesB = getSampleCount(segmentElementsB);
    uniform int   segmentSamples = segmentSamplesA + segmentSamplesB;

    startSrcA    = limitsA[taskIndex];
    startSrcB    = limitsB[taskIndex];
    uniform int endSrcA = (intervalI + 1 < segmentSamples) ? limitsA[taskIndex + 1] : segmentElementsA;
    uniform int endSrcB = (intervalI + 1 < segmentSamples) ? limitsB[taskIndex + 1] : segmentElementsB;
    lenSrcA      = endSrcA - startSrcA;
    lenSrcB      = endSrcB - startSrcB;
    startDstA    = startSrcA + startSrcB;
    startDstB    = startDstA + lenSrcA;
  }

  //Load main input data

  if (programIndex < lenSrcA)
  {
    s_key[programIndex +             0] = srcKey[0 + startSrcA + programIndex];
    s_val[programIndex +             0] = srcVal[0 + startSrcA + programIndex];
  }

  if (programIndex < lenSrcB)
  {
    s_key[programIndex + SAMPLE_STRIDE] = srcKey[stride + startSrcB + programIndex];
    s_val[programIndex + SAMPLE_STRIDE] = srcVal[stride + startSrcB + programIndex];
  }

  //Merge data in shared memory
  merge(
      s_key,
      s_val,
      s_key + 0,
      s_val + 0,
      s_key + SAMPLE_STRIDE,
      s_val + SAMPLE_STRIDE,
      lenSrcA, SAMPLE_STRIDE,
      lenSrcB, SAMPLE_STRIDE
      );

  //Store merged data

  assert(startDstA < N);
  assert(startDstB < N);
  if (programIndex < lenSrcA)
  {
    dstKey[startDstA + programIndex] = s_key[programIndex];
    dstVal[startDstA + programIndex] = s_val[programIndex];
  }

  if (programIndex < lenSrcB)
  {
    dstKey[startDstB + programIndex] = s_key[lenSrcA + programIndex];
    dstVal[startDstB + programIndex] = s_val[lenSrcA + programIndex];
  }
}
static inline
void mergeElementaryIntervals(
    uniform int dstKey[],
    uniform int dstVal[],
    uniform int srcKey[],
    uniform int srcVal[],
    uniform int limitsA[],
    uniform int limitsB[],
    uniform int stride,
    uniform int N)
{
  const uniform int lastSegmentElements = N % (2 * stride);
  const uniform int mergePairs = (lastSegmentElements > stride) ? getSampleCount(N) : (N - lastSegmentElements) / SAMPLE_STRIDE;


  launch [mergePairs] mergeElementaryIntervalsKernel(
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

static uniform int * uniform memPool = NULL;
static uniform int * uniform ranksA;
static uniform int * uniform ranksB;
static uniform int * uniform limitsA;
static uniform int * uniform limitsB;
static uniform int MAX_SAMPLE_COUNT = 0;

export 
void openMergeSort()
{
  MAX_SAMPLE_COUNT = 32 * 131072 / programCount;
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
void copyKernel(uniform int dst[], uniform int src[], uniform int size)
{
  assert(size == 0);
  foreach (i = 0 ... size)
    dst[i] = src[i];
}

export 
void mergeSort(
    uniform int dstKey[],
    uniform int dstVal[],
    uniform int bufKey[],
    uniform int bufVal[],
    uniform int srcKey[],
    uniform int srcVal[],
    uniform int N)
{
  uniform int stageCount = 0;
  for (uniform int stride = 2*programCount; stride < N; stride <<= 1, stageCount++);

  uniform int * uniform iKey, * uniform oKey;
  uniform int * uniform iVal, * uniform oVal;

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
  mergeSortGang(iKey, iVal, srcKey, srcVal, N/(2*programCount));

  for (uniform int stride = 2*programCount; stride < N; stride <<= 1)
  {
    print ("stride= % N= % \n", stride, N);
    uniform int lastSegmentElements = N % (2 * stride);

    //Find sample ranks and prepare for limiters merge
    generateSampleRanks(ranksA, ranksB, iKey, stride, N);

    //Merge ranks and indices
    mergeRanksAndIndices(limitsA, limitsB, ranksA, ranksB, stride, N);
        
    //Merge elementary intervals
    mergeElementaryIntervals(oKey, oVal, iKey, iVal, limitsA, limitsB, stride, N);

    if (lastSegmentElements <= stride)
    {
      //Last merge segment consists of a single array which just needs to be passed through
      copyKernel(oKey + (N - lastSegmentElements), iKey + (N - lastSegmentElements), lastSegmentElements);
      copyKernel(oVal + (N - lastSegmentElements), iVal + (N - lastSegmentElements), lastSegmentElements);
    }

    {
      uniform int * uniform tmpKey = iKey;
      iKey = oKey;
      oKey = tmpKey;
    }

    {
      uniform int * uniform tmpVal = iVal;
      iVal = oVal;
      oVal = tmpVal;
    }
  }
}
