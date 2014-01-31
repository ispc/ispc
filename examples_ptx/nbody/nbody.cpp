#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <iomanip>
#include "../timing.h"
#include "../ispc_malloc.h"
#include "nbody_ispc.h"
#include "plummer.h"

#include "realType.h"

int main (int argc, char *argv[])
{
  int i, j, n = argc == 1 ? 2048: atoi(argv[1]), m = n < 100 ? 1 : 50, l = n < 100 ? n : RAND_MAX;
  double tISPC1 = 0.0, tISPC2 = 0.0, tSerial = 0.0;

  printf(" nbodies= %d\n", n);

  Plummer plummer(n);

  real *posx = new real[n];
  real *posy = new real[n];
  real *posz = new real[n];
  real *velx = new real[n];
  real *vely = new real[n];
  real *velz = new real[n];
  real *mass = new real[n];

#pragma omp parallel for
  for (int i = 0; i < n; i++)
  {
    posx[i] = plummer.pos[i].x;
    posy[i] = plummer.pos[i].y;
    posz[i] = plummer.pos[i].z;
    velx[i] = plummer.vel[i].x;
    vely[i] = plummer.vel[i].y;
    velz[i] = plummer.vel[i].z;
    mass[i] = plummer.mass[i];
  }

  ispcSetMallocHeapLimit(1024*1024*1024);
  ispc::openNbody(n);

  const int nSteps = 1;
  const real dt = 0;
  tISPC2 = 1e30;
  for (i = 0; i < m; i ++)
  {
    reset_and_start_timer();
    ispc::nbodyIntegrate(
        nSteps, n, dt,
        posx, posy, posz, mass,
        velx, vely, velz,
        NULL);
    tISPC2 = get_elapsed_msec();
    fprintf(stderr, " %d iterations took %g sec; perf= %g GFlops\n",
        nSteps, tISPC2/1e3,
        nSteps * 20.0*n*n/(tISPC2/1e3)/1e9);
  }

  ispc::closeNbody();

  delete posx;
  delete posy;
  delete posz;
  delete velx;
  delete vely;
  delete velz;
  delete mass;


  return 0;
}
