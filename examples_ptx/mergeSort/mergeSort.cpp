#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <iomanip>
#include "../timing.h"
#include "../ispc_malloc.h"
#include "mergeSort_ispc.h"

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

struct Key
{
  int key, val;
};


int main (int argc, char *argv[])
{
  int i, j, n = argc == 1 ? 1000000 : atoi(argv[1]), m = n < 100 ? 1 : 50, l = n < 100 ? n : RAND_MAX;
  double tISPC1 = 0.0, tISPC2 = 0.0, tSerial = 0.0;

  Key *keys = new Key[n];
  srand48(rtc()*65536);
#pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    keys[i].key = ((int)(drand48() * (1<<30)));
    keys[i].val = i;
  }
  std::random_shuffle(keys, keys + n);

  int *keysSrc = new int[n];
  int *valsSrc = new int[n];
  int *keysBuf = new int[n];
  int *valsBuf = new int[n];
  int *keysDst = new int[n];
  int *valsDst = new int[n];
  int *keysGld = new int [n];
  int *valsGld = new int [n];
#pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    keysSrc[i] = keys[i].key;
    valsSrc[i] = keys[i].val;

    keysGld[i] = keysSrc[i];
    valsGld[i] = valsSrc[i];
  }
  delete keys;
    
  ispcSetMallocHeapLimit(1024*1024*1024);

  ispc::openMergeSort();

  tISPC2 = 1e30;
  for (i = 0; i < m; i ++)
  {
    ispcMemcpy(keysSrc, keysGld, n*sizeof(Key));
    ispcMemcpy(valsSrc, keysGld, n*sizeof(Key));

    reset_and_start_timer();
    ispc::mergeSort(keysDst, valsDst, keysBuf, valsBuf, keysSrc, valsSrc, n);
    tISPC2 = std::min(tISPC2, get_elapsed_msec());

    if (argc != 3)
        progressbar (i, m);
  }

  ispc::closeMergeSort();

  printf("[sort ispc + tasks]:\t[%.3f] msec [%.3f Mpair/s]\n", tISPC2, 1.0e-3*n/tISPC2);

  std::sort(keysGld, keysGld + n);
  for (int i = 0; i < n; i++)
    assert(keysDst[i] == keysGld[i]);

  delete keysSrc;
  delete valsSrc;
  delete keysDst;
  delete valsDst;
  delete keysBuf;
  delete valsBuf;
  delete keysGld;
  delete valsGld;

  return 0;
}
