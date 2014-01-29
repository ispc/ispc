#define SAMPLE_STRIDE programCount

static inline
int iDivUp(int a, int b)
{
  int div = a/b;
  return ((a % b) == 0) ? div : (div + 1);
}

static inline
uniform int iDivUp(uniform int a, uniform int b)
{
  uniform int div = a/b;
  return ((a % b) == 0) ? div : (div + 1);
}

static inline
int getSampleCount(int dividend)
{
  return iDivUp(dividend, SAMPLE_STRIDE);
}

  static inline
uniform int getSampleCount(uniform int dividend)
{
  return iDivUp(dividend, SAMPLE_STRIDE);
}

#define W (/*sizeof(int)=*/4 * 8)
static inline
int nextPowerOfTwo(int x)
{
  /*
     --x;
     x |= x >> 1;
     x |= x >> 2;
     x |= x >> 4;
     x |= x >> 8;
     x |= x >> 16;
     return ++x;
     */
  return 1U << (W - count_leading_zeros(x - 1));
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
void mergeSortSharedKernel(
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

////////////////////////////////////////////////////////////////////////////////
// Merge step 1: generate sample ranks
////////////////////////////////////////////////////////////////////////////////
task
void generateSampleRanksKernel(
    uniform int in_ranksA[],
    uniform int in_ranksB[],
    uniform int in_srcKey[],
    const uniform int stride,
    const uniform int N,
    const int totalProgramCount)
{
    const int pos = taskIndex * programCount + programIndex;

    if (pos >= totalProgramCount)
      return;

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

    if (pos >= totalProgramCount)
      return;

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
    uniform int N
)
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
      uniform int endSrcA = (intervalI + 1 < segmentSamples) ? limitsA[taskIndex+ 1] : segmentElementsA;
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
