#include "realType.h"
#include "cuda_helpers.cuh"
#include <cassert>

#define uniform

__device__ static uniform real * uniform accx = NULL;
__device__ static uniform real * uniform accy;
__device__ static uniform real * uniform accz;
__device__ static uniform real * uniform gpotList;

__global__
void openNbody___export(const uniform int n)
{
  assert(accx == NULL);
  accx = uniform new uniform real[n];
  accy = uniform new uniform real[n];
  accz = uniform new uniform real[n];
  gpotList = uniform new uniform real[n];
}
extern "C"
void openNbody(int n)
{
  openNbody___export<<<1,1>>>(n);
}

__global__
void closeNbody___export()
{
  assert(accx != NULL);
  delete accx;
  delete accy;
  delete accz;
  delete gpotList;
}
extern "C"
void closeNbody()
{
  closeNbody___export<<<1,1>>>();
}



__global__
void computeForces(
    uniform int  nbodies,
    uniform real posx[],
    uniform real posy[],
    uniform real posz[],
    uniform real mass[])
{
  const uniform int blkIdx = taskIndex;
  const uniform int blkDim = (nbodies + taskCount - 1)/taskCount;
  const uniform int blkBeg =     blkIdx * blkDim;
  const uniform int blkEnd = min(blkBeg + blkDim, nbodies);

  for (int i = programIndex + blkBeg; i < blkEnd; i += programCount)
    if (i < blkEnd)
    {
      const real iposx = posx[i];
      const real iposy = posy[i];
      const real iposz = posz[i];
      real iaccx = 0;
      real iaccy = 0;
      real iaccz = 0;
      real igpot = 0;
#if 0
      for (uniform int j = 0; j < nbodies; j++)
      {
        const real jposx = posx[j];
        const real jposy = posy[j];
        const real jposz = posz[j];
        const real jmass = mass[j];
        const real    dx  = jposx - iposx; 
        const real    dy  = jposy - iposy; 
        const real    dz  = jposz - iposz; 
        const real    r2  = dx*dx + dy*dy + dz*dz; 
        const real  rinv  = r2; // > 0.0 ? rsqrt((float)r2) : 0; 
        const real mrinv  = -jmass * rinv; 
        const real mrinv3 = mrinv * rinv*rinv; 
        iaccx += mrinv3 * dx; 
        iaccy += mrinv3 * dy; 
        iaccz += mrinv3 * dz; 
        igpot += mrinv; 
      }
#else
      for (uniform int j = 0; j < nbodies; j += programCount)
      {
#if 1
        __shared__ real shdata[4][programCount*4];
        real (* shmem)[programCount] = (real (*)[programCount])shdata[warpIdx];
        shmem[0][programIndex] = posx[j+programIndex];
        shmem[1][programIndex] = posy[j+programIndex];
        shmem[2][programIndex] = posz[j+programIndex];
        shmem[3][programIndex] = mass[j+programIndex];
#else
      const real jPosx = posx[j+programIndex];
      const real jPosy = posy[j+programIndex];
      const real jPosz = posz[j+programIndex];
      const real jMass = mass[j+programIndex];
#endif

#pragma unroll 1
        for (int jb = 0; jb < programCount; jb++)
        {
#if 1
          const real jposx = shmem[0][jb];
          const real jposy = shmem[1][jb];
          const real jposz = shmem[2][jb];
          const real jmass = shmem[3][jb];
#else
          const real jposx = broadcast(jPosx, jb);
          const real jposy = broadcast(jPosy, jb);
          const real jposz = broadcast(jPosz, jb);
          const real jmass = broadcast(jMass, jb);
#endif
          const real    dx  = jposx - iposx; 
          const real    dy  = jposy - iposy; 
          const real    dz  = jposz - iposz; 
          const real    r2  = dx*dx + dy*dy + dz*dz; 
          const real  rinv  = r2 > 0.0 ? rsqrt((float)r2) : 0; 
          const real mrinv  = -jmass * rinv; 
          const real mrinv3 = mrinv * rinv*rinv; 
          iaccx += mrinv3 * dx; 
          iaccy += mrinv3 * dy; 
          iaccz += mrinv3 * dz; 
          igpot += mrinv; 
        }
      }
#endif
      accx[i]  = iaccx;
      accy[i]  = iaccy;
      accz[i]  = iaccz;
    }
}

__global__
void updatePositions(
    uniform int  nbodies,
    uniform real posx[],
    uniform real posy[],
    uniform real posz[],
    uniform real velx[],
    uniform real vely[],
    uniform real velz[],
    uniform real dt)
{
  const uniform int blkIdx = taskIndex;
  const uniform int blkDim = (nbodies + taskCount - 1)/taskCount;
  const uniform int blkBeg =     blkIdx * blkDim;
  const uniform int blkEnd = min(blkBeg + blkDim, nbodies);

  for (int i = programIndex + blkBeg; i < blkEnd; i += programCount)
    if (i < blkEnd)
    {
      posx[i] += dt*velx[i];
      posy[i] += dt*vely[i];
      posz[i] += dt*velz[i];
    }
}

__global__
void updateVelocities(
    uniform int  nbodies,
    uniform real velx[],
    uniform real vely[],
    uniform real velz[],
    uniform real dt)
{
  const uniform int blkIdx = taskIndex;
  const uniform int blkDim = (nbodies + taskCount - 1)/taskCount;
  const uniform int blkBeg =     blkIdx * blkDim;
  const uniform int blkEnd = min(blkBeg + blkDim, nbodies);

  for (int i = programIndex + blkBeg; i < blkEnd; i += programCount)
    if (i < blkEnd)
    {
      velx[i] += dt*accx[i];
      vely[i] += dt*accy[i];
      velz[i] += dt*accz[i];
    }
}

__global__
void nbodyIntegrate___export(
    uniform int  nSteps,
    uniform int  nbodies,
    uniform real dt,
    uniform real posx[],
    uniform real posy[],
    uniform real posz[],
    uniform real mass[],
    uniform real velx[],
    uniform real vely[],
    uniform real velz[],
    uniform real energies[])
{
  uniform int nTasks ;
  nTasks = (nbodies+1*programCount - 1)/(1*programCount);

  for (uniform int step = 0; step < nSteps; step++)
  { 
    //    launch (nTasks,1,1, updatePositions)(nbodies, posx, posy, posz, velx, vely, velz,dt);
    //   sync;
    launch (nTasks,1,1, computeForces)(nbodies, posx, posy, posz, mass);
    sync;
    // launch (nTasks,1,1, updateVelocities)(nbodies, posx, posy, posz, dt);
    //sync;
  }

#if 0
  if (energies != NULL)
  {
    real gpotLoc = 0;
    foreach (i = 0 ... nTasks)
      gpotLoc += gpotList[i];
    energies[0] = reduce_add(gpotLoc);
  }
#endif
}


extern "C"
void nbodyIntegrate(
    uniform int  nSteps,
    uniform int  nbodies,
    uniform real dt,
    uniform real posx[],
    uniform real posy[],
    uniform real posz[],
    uniform real mass[],
    uniform real velx[],
    uniform real vely[],
    uniform real velz[],
    uniform real energies[])
{
  nbodyIntegrate___export<<<1,32>>>(
      nSteps,
      nbodies,
      dt,
      posx,
      posy,
      posz,
      mass,
      velx,
      vely,
      velz,
      energies);
  sync;
}
