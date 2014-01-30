#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <iomanip>
#include "../timing.h"
#include "../ispc_malloc.h"
#include "bandwidthTest_ispc.h"

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

typedef double T;

int main (int argc, char *argv[])
{
  int i, j, n = argc == 1 ? 32*1024*1024: atoi(argv[1]), m = n < 100 ? 1 : 50, l = n < 100 ? n : RAND_MAX;
  double tISPC1 = 0.0, tISPC2 = 0.0, tSerial = 0.0;

  T *src = new T[n];
  T *src_orig = new T[n];
  T *dst = new T[n];
#pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    src[i] = i;
    src_orig[i] = src[i];
    dst[i] = -1;
  }
    
  tISPC2 = 1e30;
  for (i = 0; i < m; i ++)
  {
    ispcMemcpy(src, src_orig, n*sizeof(T));

    reset_and_start_timer();
    ispc::copy(dst, src, n);
    tISPC2 = std::min(tISPC2, get_elapsed_msec());

    if (argc != 3)
        progressbar (i, m);
  }


  printf("[bandwidthTest ispc + tasks]:\t[%.3f] msec [%.3f MB/s]\n",
      tISPC2, 1.0e-3*2*sizeof(T)*n/tISPC2);


  for (int i = 0; i < n; i++)
    assert(src[i] == dst[i]);

  delete src;
  delete dst;

  return 0;
}
