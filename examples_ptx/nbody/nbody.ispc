#include "realType.h"


static uniform real * uniform accx = NULL;
static uniform real * uniform accy;
static uniform real * uniform accz;
static uniform real * uniform gpotList;

export
void openNbody(const uniform int n)
{
  assert(accx == NULL);
  accx = uniform new uniform real[n];
  accy = uniform new uniform real[n];
  accz = uniform new uniform real[n];
  gpotList = uniform new uniform real[n];
}

export 
void closeNbody()
{
  assert(accx != NULL);
  delete accx;
  delete accy;
  delete accz;
  delete gpotList;
}

  static inline
real rsqrt_real(real r2)
{
#if 0
  return r2> (real)0.0 ? (real)1.0d0/sqrt(r2) : 0; 
#else
  return r2> (real)0.0 ? rsqrt((float)r2) : 0; 
#endif
}

uniform int nn = programCount;


task 
void computeForces(
    uniform int  nbodies,
    uniform real posx[],
    uniform real posy[],
    uniform real posz[],
    uniform real mass[])
{
  const uniform int blockIdx = taskIndex;
  const uniform int blockDim = (nbodies + taskCount - 1)/taskCount;
  const uniform int blockBeg =     blockIdx * blockDim;
  const uniform int blockEnd = min(blockBeg + blockDim, nbodies);
      uniform real shmem[4][programCount];

  //real gpotLoc = 0;
  foreach (i = blockBeg ... blockEnd)
  {
    const real iposx = posx[i];
    const real iposy = posy[i];
    const real iposz = posz[i];
    real iaccx = 0;
    real iaccy = 0;
    real iaccz = 0;
    real igpot = 0;
#ifndef __NVPTX__
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
      const real  rinv  = rsqrt_real(r2);
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
      for (uniform int jb = 0; jb < programCount; jb++)
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
#if 0 /* faster when used with full dp */
        const real rinv  = r2> (real)0.0 ? (real)1.0d0/sqrt(r2) : 0; 
#else
        const real rinv  = rsqrt_real(r2);
#endif
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
//    gpotLoc += igpot;
  }
//  gpotList[taskIndex] = reduce_add(gpotLoc);
}

task
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
  const uniform int blockIdx = taskIndex;
  const uniform int blockDim = (nbodies + taskCount - 1)/taskCount;
  const uniform int blockBeg =     blockIdx * blockDim;
  const uniform int blockEnd = min(blockBeg + blockDim, nbodies);

  foreach (i = blockBeg ... blockEnd)
  {
    posx[i] += dt*velx[i];
    posy[i] += dt*vely[i];
    posz[i] += dt*velz[i];
  }
}

task
void updateVelocities(
    uniform int  nbodies,
    uniform real velx[],
    uniform real vely[],
    uniform real velz[],
    uniform real dt)
{
  const uniform int blockIdx = taskIndex;
  const uniform int blockDim = (nbodies + taskCount - 1)/taskCount;
  const uniform int blockBeg =     blockIdx * blockDim;
  const uniform int blockEnd = min(blockBeg + blockDim, nbodies);

  foreach (i = blockBeg ... blockEnd)
  {
    velx[i] += dt*accx[i];
    vely[i] += dt*accy[i];
    velz[i] += dt*accz[i];
  }
}

export
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
  uniform int nTasks = num_cores()*4;
#ifdef __NVPTX__
  nTasks = (nbodies + programCount - 1)/(programCount);
#endif
  assert((nbodies % nTasks) == 0);

  for (uniform int step = 0; step < nSteps; step++)
  { 
//    launch [nTasks] updatePositions(nbodies, posx, posy, posz, velx, vely, velz,dt);
 //   sync;
    launch [nTasks] computeForces(nbodies, posx, posy, posz, mass);
    sync;
  //  launch [nTasks] updateVelocities(nbodies, posx, posy, posz, dt);
   // sync;
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


