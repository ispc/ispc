#define SIZE_LIMIT (2*programCount)

static inline void Comparator(
    int &keyA,
    int &valA,
    int &keyB,
    int &valB,
    const int dir)
{
  if ((keyA > keyB) == dir)
  {
    int t;
    t = keyA; keyA = keyB; keyB = t;
    t = valA; valA = valB; valB = t;
  }
}
                                                                                                     
////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into local memory
////////////////////////////////////////////////////////////////////////////////
task
void bitonicSortLocal(
    uniform int dstKey[],
    uniform int dstVal[],
    uniform int srcKey[],
    uniform int srcVal[],
    const uniform int arrayLength,
    const uniform int dir)
{
  uniform int l_key[SIZE_LIMIT];
  uniform int l_val[SIZE_LIMIT];

  //Offset to the beginning of subbatch and load data
  const int offset = taskIndex0 * SIZE_LIMIT + programIndex;
  l_key[programIndex +              0] = srcKey[offset];
  l_val[programIndex +              0] = srcVal[offset];
  l_key[programIndex + (SIZE_LIMIT/2)] = srcKey[offset +(SIZE_LIMIT/2)];
  l_val[programIndex + (SIZE_LIMIT/2)] = srcVal[offset +(SIZE_LIMIT/2)];

  for (uniform int size = 2; size < arrayLength; size <<= 1)
  {
    //Bitonic merge
    const int ddd = dir ^ ( (programIndex & (size / 2)) != 0 );
    for (uniform int stride = size / 2; stride > 0; stride >>= 1)
    {
      const int pos = 2 * programIndex - (programIndex & (stride - 1));
      int key_a = l_key[pos];
      int val_a = l_val[pos];
      int key_b = l_key[pos + stride];
      int val_b = l_val[pos + stride];
      Comparator(key_a, val_a, key_b, val_b, ddd);
      l_key[pos] = key_a;
      l_val[pos] = val_a;
      l_key[pos + stride] = key_b;
      l_val[pos + stride] = val_b;
    }
  }

  //ddd == dir for the last bitonic merge step
  {
    for (int stride = arrayLength / 2; stride > 0; stride >>= 1)
    {
      const int pos = 2 * programIndex - (programIndex & (stride - 1));
      int key_a = l_key[pos];
      int val_a = l_val[pos];
      int key_b = l_key[pos + stride];
      int val_b = l_val[pos + stride];
      Comparator(key_a, val_a, key_b, val_b, dir);
      l_key[pos] = key_a;
      l_val[pos] = val_a;
      l_key[pos + stride] = key_b;
      l_val[pos + stride] = val_b;
    }
  }

  dstKey[offset]                 = l_key[programIndex +              0];
  dstVal[offset]                 = l_val[programIndex +              0];
  dstKey[offset +(SIZE_LIMIT/2)] = l_key[programIndex + (SIZE_LIMIT/2)];
  dstVal[offset +(SIZE_LIMIT/2)] = l_val[programIndex + (SIZE_LIMIT/2)];
}

////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into local memory)
////////////////////////////////////////////////////////////////////////////////
//Bottom-level bitonic sort
//Almost the same as bitonicSortLocal with the only exception
//of even / odd subarrays (of LOCAL_SIZE_LIMIT points) being
//sorted in opposite directions
task
void bitonicSortLocal1(
    uniform int dstKey[],
    uniform int dstVal[],
    uniform int srcKey[],
    uniform int srcVal[])
{
  uniform int l_key[SIZE_LIMIT];
  uniform int l_val[SIZE_LIMIT];

  //Offset to the beginning of subarray and load data
  const int offset = taskIndex0 * SIZE_LIMIT + programIndex;
  l_key[programIndex +              0] = srcKey[offset];
  l_val[programIndex +              0] = srcVal[offset];
  l_key[programIndex + (SIZE_LIMIT/2)] = srcKey[offset + (SIZE_LIMIT/2)];
  l_val[programIndex + (SIZE_LIMIT/2)] = srcVal[offset + (SIZE_LIMIT/2)];

  for (int size = 2; size < SIZE_LIMIT; size <<= 1)
  {
    //Bitonic merge
    const int ddd = (programIndex & (size / 2)) != 0;
    for (int stride = size / 2; stride > 0; stride >>= 1)
    {
      const int pos = 2 * programIndex - (programIndex & (stride - 1));
      int key_a = l_key[pos];
      int val_a = l_val[pos];
      int key_b = l_key[pos + stride];
      int val_b = l_val[pos + stride];
      Comparator(key_a, val_a, key_b, val_b, ddd);
      l_key[pos] = key_a;
      l_val[pos] = val_a;
      l_key[pos + stride] = key_b;
      l_val[pos + stride] = val_b;
    }
  }

  //Odd / even arrays of LOCAL_SIZE_LIMIT elements
  //sorted in opposite directions
  {
    const int ddd = taskIndex0 & 1;
    for (int stride = SIZE_LIMIT/2; stride > 0; stride >>= 1)
    {
      const int pos = 2 * programIndex - (programIndex & (stride - 1));
      int key_a = l_key[pos];
      int val_a = l_val[pos];
      int key_b = l_key[pos + stride];
      int val_b = l_val[pos + stride];
      Comparator(key_a, val_a, key_b, val_b, ddd);
      l_key[pos] = key_a;
      l_val[pos] = val_a;
      l_key[pos + stride] = key_b;
      l_val[pos + stride] = val_b;
    }
  }

  dstKey[offset]                  = l_key[programIndex +              0];
  dstVal[offset]                  = l_val[programIndex +              0];
  dstKey[offset + (SIZE_LIMIT/2)] = l_key[programIndex + (SIZE_LIMIT/2)];
  dstVal[offset + (SIZE_LIMIT/2)] = l_val[programIndex + (SIZE_LIMIT/2)];
}

//Bitonic merge iteration for 'stride' >= LOCAL_SIZE_LIMIT
task
void bitonicMergeGlobal(
    uniform int dstKey[],
    uniform int dstVal[],
    uniform int srcKey[],
    uniform int srcVal[],
    const uniform int arrayLength,
    const uniform int size,
    const uniform int stride,
    const uniform int dir)
{
  const int global_comparatorI = taskIndex0*programCount + programIndex;
  const int        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

  //Bitonic merge
  const int ddd = dir ^ ( (comparatorI & (size / 2)) != 0 );
  const int pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

  int keyA = srcKey[pos +      0];
  int valA = srcVal[pos +      0];
  int keyB = srcKey[pos + stride];
  int valB = srcVal[pos + stride];

  Comparator(
      keyA, valA,
      keyB, valB,
      ddd
      );

  dstKey[pos +      0] = keyA;
  dstVal[pos +      0] = valA;
  dstKey[pos + stride] = keyB;
  dstVal[pos + stride] = valB;
}

//Combined bitonic merge steps for
//'size' > LOCAL_SIZE_LIMIT and 'stride' = [1 .. LOCAL_SIZE_LIMIT / 2]
task
void bitonicMergeLocal(
    uniform int dstKey[],
    uniform int dstVal[],
    uniform int srcKey[],
    uniform int srcVal[],
    const uniform int arrayLength,
    const uniform int size,
    const uniform int dir)
{
  uniform int l_key[SIZE_LIMIT];
  uniform int l_val[SIZE_LIMIT];

  const int offset = taskIndex0 * SIZE_LIMIT + programIndex;
  l_key[programIndex +              0] = srcKey[offset];
  l_val[programIndex +              0] = srcVal[offset];
  l_key[programIndex + (SIZE_LIMIT/2)] = srcKey[offset +(SIZE_LIMIT/2)];
  l_val[programIndex + (SIZE_LIMIT/2)] = srcVal[offset +(SIZE_LIMIT/2)];

  //Bitonic merge
  const int   global_id = taskIndex0*programCount + programIndex;
  const int comparatorI = global_id & ((arrayLength / 2) - 1);
  const int         ddd = dir ^ ( (comparatorI & (size / 2)) != 0 );
  for (int stride = SIZE_LIMIT / 2; stride > 0; stride >>= 1)
  {
      const int pos = 2 * programIndex - (programIndex & (stride - 1));
      int key_a = l_key[pos];
      int val_a = l_val[pos];
      int key_b = l_key[pos + stride];
      int val_b = l_val[pos + stride];
      Comparator(key_a, val_a, key_b, val_b, ddd);
      l_key[pos] = key_a;
      l_val[pos] = val_a;
      l_key[pos + stride] = key_b;
      l_val[pos + stride] = val_b;
  }

  dstKey[offset]                 = l_key[programIndex +              0];
  dstVal[offset]                 = l_val[programIndex +              0];
  dstKey[offset +(SIZE_LIMIT/2)] = l_key[programIndex + (SIZE_LIMIT/2)];
  dstVal[offset +(SIZE_LIMIT/2)] = l_val[programIndex + (SIZE_LIMIT/2)];
}

static inline int factorRadix2(int &log2L, int L){
  if(!L){
    log2L = 0;
    return 0;
  }else{
    int val;
    for(val = 0; (L & 1) == 0; L >>= 1, val++);
    log2L = val;
    return L;
  }
}

export
void bitoniSort(
    uniform int dstKey[],
    uniform int dstVal[],
    uniform int srcKey[],
    uniform int srcVal[],
    const uniform int batchSize,
    const uniform int arrayLength,
    const uniform int dir)
{
  //Nothing to sort
  if (arrayLength < 2)
    return;
    
  //Only power-of-two array lengths are supported by this implementation
  int log2L;
  const int factorizationRemainder = factorRadix2(log2L, arrayLength);
  assert( factorizationRemainder == 1 );
    
  const uniform int  blockCount = batchSize * arrayLength / SIZE_LIMIT;
  const uniform int threadCount = SIZE_LIMIT / 2;

  if (arrayLength <= SIZE_LIMIT)
  {
    assert( (batchSize * arrayLength) % SIZE_LIMIT == 0 );
    launch [blockCount] bitonicSortLocal(dstKey, dstVal, srcKey, srcVal, arrayLength, dir);
    sync;
  }
  else
  {
    launch [blockCount] bitonicSortLocal1(dstKey, dstVal, srcKey, srcVal);
    sync;

    for(uniform int size = 2 * SIZE_LIMIT; size <= arrayLength; size <<= 1)
      for(uniform int stride = size / 2; stride > 0; stride >>= 1)
        if (stride >= SIZE_LIMIT)
        {
          launch [blockCount] bitonicMergeGlobal(dstKey, dstVal, dstKey, dstVal, arrayLength, size, stride, dir);
          sync;
        }
        else
        {
          launch [blockCount] bitonicMergeLocal(dstKey, dstVal, dstKey, dstVal, arrayLength, size, dir);
          sync;
        }
  }
}
