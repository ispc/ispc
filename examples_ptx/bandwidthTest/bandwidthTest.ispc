typedef double T;

task
void copyKernel(
    uniform T dst[],
    uniform T src[],
    uniform int nTotal)
{
  const uniform int blockIdx = taskIndex;
  const uniform int blockDim = (nTotal + taskCount - 1) / taskCount;
  const uniform int blockBeg =     blockIdx * blockDim;
  const uniform int blockEnd = min(blockBeg + blockDim, nTotal);

  foreach (i = blockBeg ... blockEnd)
    src[i] = dst[i];
}

export 
void copy(
    uniform T dst[],
    uniform T src[],
    uniform int nTotal)
{
  uniform int nTask = num_cores() * 4;
#ifdef __NVPTX__
  nTask  = nTotal/(8*programCount);
#endif

  launch [nTask] copyKernel(dst, src, nTotal);
}
