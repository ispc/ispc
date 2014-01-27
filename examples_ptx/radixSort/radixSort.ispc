#if 1
struct int2 {  int x,y; };
struct int4 {  int x,y,z,w; };
#else
typedef int<2> int2;
typedef int<4> int4;
#endif

static int4 scan4(const int4 idata)
{    
  const int idx = programIndex;

  int4 val4 = idata;
  int sum[3];
  sum[0] = val4.x;
  sum[1] = val4.y + sum[0];
  sum[2] = val4.z + sum[1];
    
  int val = val4.w + sum[2];
  val = exclusive_scan_add(val);

  val4.x = val;
  val4.y = val + sum[0];
  val4.z = val + sum[1];
  val4.w = val + sum[2];

  return val4;
}

static int4 rank4(int4 preds)
{
	const         int localId   = programIndex;
	const uniform int localSize = programCount;

	const int4 address = scan4(preds);
	
  const int numtrue = broadcast(address.w + preds.w, localSize-1);
	
	int4 rank;
	const int idx = localId*4;
	rank.x = (preds.x) ? address.x : numtrue + idx - address.x;
	rank.y = (preds.y) ? address.y : numtrue + idx + 1 - address.y;
	rank.z = (preds.z) ? address.z : numtrue + idx + 2 - address.z;
	rank.w = (preds.w) ? address.w : numtrue + idx + 3 - address.w;
	
	return rank;
}

static void radixSortBlockKeysOnly(int4 &key_inout, uniform int nbits, uniform int startbit)
{
  const         int localId   = programIndex;
  const uniform int localSize = programCount;

  uniform int sMem[programCount*4];

  int4 key = key_inout;  
	for (uniform int shift = startbit; shift < (startbit + nbits); ++shift)
  {
    int4 lsb;
    lsb.x = !((key.x >> shift) & 0x1);
    lsb.y = !((key.y >> shift) & 0x1);
    lsb.z = !((key.z >> shift) & 0x1);
    lsb.w = !((key.w >> shift) & 0x1);

    const int4 r = rank4(lsb);

    // This arithmetic strides the ranks across 4 CTA_SIZE regions
    sMem[(r.x & 3) * localSize + (r.x >> 2)] = key.x;
    sMem[(r.y & 3) * localSize + (r.y >> 2)] = key.y;
    sMem[(r.z & 3) * localSize + (r.z >> 2)] = key.z;
    sMem[(r.w & 3) * localSize + (r.w >> 2)] = key.w;

    // The above allows us to read without 4-way bank conflicts:
    key.x = sMem[localId                ];
    key.y = sMem[localId +     localSize];
    key.z = sMem[localId + 2 * localSize];
    key.w = sMem[localId + 3 * localSize];
  }
}

task void radixSortBlocksKeysOnly(
    uniform int keysIn[],
    uniform int keysOut[],
    uniform int nbits,
    uniform int startbit,
    uniform int numElements, 
    uniform int totalBlocks)
{
	const int globalId = taskIndex * programCount + programIndex;

	int4 key;
	key.x = keysIn[4*globalId + 0];
	key.y = keysIn[4*globalId + 1];
	key.z = keysIn[4*globalId + 2];
	key.w = keysIn[4*globalId + 3];
	
	radixSortBlockKeysOnly(key, nbits, startbit);
	
	keysOut[4*globalId+0] = key.x;
	keysOut[4*globalId+1] = key.y;
	keysOut[4*globalId+2] = key.z;
	keysOut[4*globalId+3] = key.w;
}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each 
// block counts the number of keys that fall into each radix in the group, and 
// finds the starting offset of each radix in the block.  It then writes the radix 
// counts to the counters array, and the starting offsets to the blockOffsets array.
//
// Template parameters are used to generate efficient code for various special cases
// For example, we have to handle arrays that are a multiple of the block size 
// (fullBlocks) differently than arrays that are not. "loop" is used when persistent 
// CTAs are used. 
//
// By persistent CTAs we mean that we launch only as many thread blocks as can 
// be resident in the GPU and no more, rather than launching as many threads as
// we have elements. Persistent CTAs loop over blocks of elements until all work
// is complete.  This can be faster in some cases.  In our tests it is faster
// for large sorts (and the threshold is higher on compute version 1.1 and earlier
// GPUs than it is on compute version 1.2 GPUs.
//                                
//----------------------------------------------------------------------------
task void findRadixOffsets(
    uniform int keys[],
    uniform int counters[],
    uniform int blockOffsets[],
    uniform int startbit,
    uniform int numElements,
    uniform int totalBlocks)
{
  uniform int sStartPointers[16];

  const uniform int groupId   = taskIndex;
  const         int localId   = programIndex;
  const uniform int groupSize = programCount;
	const         int globalId  = taskIndex * programCount + programIndex;

  int2 radix2;

  radix2.x = keys[2*globalId + 0];
  radix2.y = keys[2*globalId + 1];

  uniform int sRadix1[4*programCount];
        
  sRadix1[2 * localId]     = (radix2.x >> startbit) & 0xF;
  sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;

  // Finds the position where the sRadix1 entries differ and stores start 
  // index for each radix.
  if(localId < 16) 
    sStartPointers[localId] = 0; 

  if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) ) 
    sStartPointers[sRadix1[localId]] = localId;

  if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1]) 
    sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;

  if(localId < 16) 
    blockOffsets[groupId*16 + localId] = sStartPointers[localId];

    // Compute the sizes of each block.
  if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) ) 
    sStartPointers[sRadix1[localId - 1]] = 
      localId - sStartPointers[sRadix1[localId - 1]];
  if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] ) 
    sStartPointers[sRadix1[localId + groupSize - 1]] = 
      localId + groupSize - sStartPointers[sRadix1[localId + groupSize - 1]];
        

  if(localId == groupSize - 1) 
    sStartPointers[sRadix1[2 * groupSize - 1]] = 
      2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];

  if(localId < 16) 
    counters[localId * totalBlocks + groupId] = sStartPointers[localId];
}

// a naive scan routine that works only for array that
// can fit into a single block, just for debugging purpose,
// not used in the sort now
task void scanNaive(
    uniform int odata[],
    uniform int idata[],
    uniform int n)
{
  if (programIndex < n)
    odata[programIndex] = exclusive_scan_add(idata[programIndex]);
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets 
// have been found. On compute version 1.1 and earlier GPUs, this code depends 
// on RadixSort::CTA_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).
// 
// On compute version 1.1 GPUs ("manualCoalesce=true") this function ensures
// that all writes are coalesced using extra work in the kernel.  On later
// GPUs coalescing rules have been relaxed, so this extra overhead hurts 
// performance.  On these GPUs we set manualCoalesce=false and directly store
// the results.
//
// Template parameters are used to generate efficient code for various special cases
// For example, we have to handle arrays that are a multiple of the block size 
// (fullBlocks) differently than arrays that are not.  "loop" is used when persistent 
// CTAs are used. 
//
// By persistent CTAs we mean that we launch only as many thread blocks as can 
// be resident in the GPU and no more, rather than launching as many threads as
// we have elements. Persistent CTAs loop over blocks of elements until all work
// is complete.  This can be faster in some cases.  In our tests it is faster
// for large sorts (and the threshold is higher on compute version 1.1 and earlier
// GPUs than it is on compute version 1.2 GPUs.
//----------------------------------------------------------------------------
task void reorderDataKeysOnly(
    uniform int outKeys[],
    uniform int keys[],
    uniform int blockOffsets[],
    uniform int offsets[],
    uniform int sizes[],
    uniform int startbit,
    uniform int numElements,
    uniform int totalBlocks)
{
  uniform int sOffsets[16];
  uniform int sBlockOffsets[16];
  uniform int2 sKeys2[programCount];
  uniform int * uniform sKeys1 = (uniform int * uniform)&sKeys2[0];

  const uniform int   groupId = taskIndex;
  const         int  globalId = taskIndex*programCount + programIndex;
  const         int   localId = programIndex;
  const uniform int groupSize = programCount;

  sKeys2[localId].x   = keys[2*globalId + 0];
  sKeys2[localId].y   = keys[2*globalId + 1];

  if(localId < 16)  
  {
    sOffsets[localId]      = offsets[localId * totalBlocks + groupId];
    sBlockOffsets[localId] = blockOffsets[groupId * 16 + localId];
  }

  int radix = (sKeys1[localId] >> startbit) & 0xF;
  int globalOffset = sOffsets[radix] + localId - sBlockOffsets[radix];

  if (globalOffset < numElements)
    outKeys[globalOffset] = sKeys1[localId];

  radix = (sKeys1[localId + groupSize] >> startbit) & 0xF;
  globalOffset = sOffsets[radix] + localId + groupSize - sBlockOffsets[radix];

  if (globalOffset < numElements)
    outKeys[globalOffset] = sKeys1[localId + groupSize];
}
