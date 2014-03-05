#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <vector>
#include "timing.h"
#include "ispc_malloc.h"
#include "inplaceTranspose_ispc.h"
#include "typeT.h"

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

  cerr << setw(3) << (int)(ratio*100) << "% [";
  for (int x=0; x<c; x++) cerr << "=";
  for (int x=c; x<w; x++) cerr << " ";
  cerr << "]\r" << flush;
}

int main(int argc, char * argv[])
{
  int m = argc > 1 ? atoi(argv[1]) : 8;
  int n = argc > 2 ? atoi(argv[2]) : 12;
  bool verbose = argc > 3;


  fprintf(stderr, " m= %d  n= %d :: storage= %g MB\n", m, n,
      m*n*sizeof(int)*2/1e6);


  std::vector< std::pair<int,int> > A(m*n);

  int nrep = 10;
  double dt = 1e10;
  for (int r = 0; r < nrep; r++)
  {
    for (int j = 0; j < n; j++)
      for (int i = 0; i < m; i++)
        A[j*m+i] = std::make_pair(i,j);

    if (r == 0 && verbose)
    {
      fprintf(stderr, "Original: \n");
      for (int j = 0; j < n; j++)
      {
        for (int i = 0; i < m; i++)
        {
          fprintf(stderr, "(%2d,%2d) ", A[j*m+i].first, A[j*m+i].second);
        }
        fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
      for (int i = 0; i < m*n; i++)
        fprintf(stderr, "(%2d,%2d) ", A[i].first, A[i].second);
      fprintf(stderr, "\n");
      fprintf(stderr, "\n");
    }

    for (int j = 0; j < n; j++)
      for (int i = 0; i < m; i++)
        assert(A[j*m+i].first == i && A[j*m+i].second == j);

    reset_and_start_timer();
    ispc::transpose((T*)&A[0], n, m);
    const double t1 = rtc();
    dt = std::min(dt, get_elapsed_msec());
    progressbar (r, nrep);
  }
  progressbar (nrep, nrep);
  fprintf(stderr, "\n");

  if (verbose)
  {
    fprintf(stderr, "Transposed: \n");
    for (int j = 0; j < m; j++)
    {
      for (int i = 0; i < n; i++)
      {
        fprintf(stderr, "(%2d,%2d) ", A[j*n+i].first, A[j*n+i].second);
      }
      fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    for (int i = 0; i < m*n; i++)
      fprintf(stderr, "(%2d,%2d) ", A[i].first, A[i].second);
    fprintf(stderr, "\n");
    fprintf(stderr, "\n");
  }

  for (int j = 0; j < m; j++)
    for (int i = 0; i < n; i++)
      assert(A[j*n+i].first == j && A[j*n+i].second == i);

  fprintf(stderr, " tranpose done in %g msec :: BW= % GB/s\n", 
      dt , 2*m*n*sizeof(int)*2/dt*1e3/1e9);


  return 0;
}
