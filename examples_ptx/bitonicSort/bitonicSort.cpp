#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <iomanip>
#include "../timing.h"
#include "../ispc_malloc.h"
#include "bitonicSort_ispc.h"

extern void bitonicSort_serial (
    int dstKey[],
    int dstVal[],
    int srcKey[],
    int srcVal[],
    const int batchSize,
    const int arrayLength,
    const int dir);

/* progress bar by Ross Hemsley;
 * http://www.rosshemsley.co.uk/2011/02/creating-a-progress-bar-in-c-or-any-other-console-app/ */
static inline void progressbar (unsigned int x, unsigned int n, unsigned int w = 50)
{
  if (n < 100)
  {
    x *= 100/n;
    n = 100;
  }

  if ((x != n) && (x % (n/100) != 0)) return;

  using namespace std;
  float ratio  =  x/(float)n;
  int c =  ratio * w;

  cout << setw(3) << (int)(ratio*100) << "% [";
  for (int x=0; x<c; x++) cout << "=";
  for (int x=c; x<w; x++) cout << " ";
  cout << "]\r" << flush;
}

int main (int argc, char *argv[])
{
  int i, j, n = argc == 1 ? 1000000 : atoi(argv[1]), m = n < 100 ? 1 : 50, l = n < 100 ? n : RAND_MAX;
  double tISPC1 = 0.0, tISPC2 = 0.0, tSerial = 0.0;
  unsigned int *code      = new unsigned int [n];
  unsigned int *code_orig = new unsigned int [n];
  int *order = new int [n];
    
  for (j = 0; j < n; j ++) code_orig[j] = rand() % l;

  ispcSetMallocHeapLimit(1024*1024*1024);

  srand (0);

#ifndef _CUDA_
  for (i = 0; i < m; i ++)
  {
    ispcMemcpy(code, code_orig, n*sizeof(unsigned int));

    reset_and_start_timer();

    sort_ispc (n, code, order, 1);

    tISPC1 += get_elapsed_msec();

    if (argc != 3)
        progressbar (i, m);
  }

  printf("[sort ispc]:\t[%.3f] msec [%.3f Mpair/s]\n", tISPC1, 1.0e-3*n*m/tISPC1);
#endif

  srand (0);

  const int ntask = 13*8;
  for (i = 0; i < m; i ++)
  {
    ispcMemcpy(code, code_orig, n*sizeof(unsigned int));

    reset_and_start_timer();

    sort_ispc (n, code, order, ntask);

    tISPC2 += get_elapsed_msec();

    if (argc != 3)
        progressbar (i, m);
  }

  printf("[sort ispc + tasks]:\t[%.3f] msec [%.3f Mpair/s]\n", tISPC2, 1.0e-3*n*m/tISPC2);
  unsigned int *code1 =  new unsigned int [n];
  for (int i = 0; i < n; i++) 
    code1[i] = code[i];
  std::sort(code1, code1+n);
  for (int i = 0; i < n; i++)
    assert(code1[i] == code[i]);

  srand (0);

  for (i = 0; i < m; i ++)
  {
    ispcMemcpy(code, code_orig, n*sizeof(unsigned int));

    reset_and_start_timer();

    sort_serial (n, code, order);

    tSerial += get_elapsed_msec();

    if (argc != 3)
        progressbar (i, m);
  }

  printf("[sort serial]:\t\t[%.3f] msec [%.3f Mpair/s]\n", tSerial, 1.0e-3*n*m/tSerial);

#ifndef _CUDA_
  printf("\t\t\t\t(%.2fx speedup from ISPC, %.2fx speedup from ISPC + tasks)\n", tSerial/tISPC1, tSerial/tISPC2);
#else
  printf("\t\t\t\t(%.2fx speedup from ISPC + tasks)\n", tSerial/tISPC2);
#endif

  delete code;
  delete code_orig;
  delete order;
  return 0;
}
