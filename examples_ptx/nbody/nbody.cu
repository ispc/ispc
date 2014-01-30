typedef double real;


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

#if 0
  uniform real gpotLoc = 0;
  for (uniform int i = blockBeg; i < blockEnd; i++)
  {
    const real iposx = posx[i];
    const real iposy = posy[i];
    const real iposz = posz[i];
    real iaccx = 0;
    real iaccy = 0;
    real iaccz = 0;
    real igpot = 0;
    foreach (j = 0 ... nbodies)
    {
      const real jposx = posx[j];
      const real jposy = posy[j];
      const real jposz = posz[j];
      const real jmass = mass[j];
      const real    dx  = jposx - iposx;
      const real    dy  = jposy - iposy;
      const real    dz  = jposz - iposz;
      const real    r2  = dx*dx + dy*dy + dz*dz;
      const real  rinv  = r2 > 0.0d ? rsqrt((float)r2) : 0;
      const real mrinv  = -jmass * rinv;
      const real mrinv3 = mrinv * rinv*rinv;

      iaccx += mrinv3 * dx;
      iaccy += mrinv3 * dy;
      iaccz += mrinv3 * dz;
      igpot += mrinv;
    }
    accx[i]  = reduce_add(iaccx);
    accy[i]  = reduce_add(iaccy);
    accz[i]  = reduce_add(iaccz);
    gpotLoc += reduce_add(igpot);
  }
  gpotList[taskIndex] = gpotLoc;
#else
  real gpotLoc = 0;
  foreach (i = blockBeg ... blockEnd)
  {
    const real iposx = posx[i];
    const real iposy = posy[i];
    const real iposz = posz[i];
    real iaccx = 0;
    real iaccy = 0;
    real iaccz = 0;
    real igpot = 0;
    for (uniform int j = 0; j < nbodies; j += 1)
    {
#define STEP(jk) {\
      const real jposx = posx[j+jk]; \
      const real jposy = posy[j+jk]; \
      const real jposz = posz[j+jk]; \
      const real jmass = mass[j+jk]; \
      const real    dx  = jposx - iposx; \
      const real    dy  = jposy - iposy; \
      const real    dz  = jposz - iposz; \
      const real    r2  = dx*dx + dy*dy + dz*dz; \
      const real  rinv  = r2 > 0.0d ? rsqrt((float)r2) : 0; \
      const real mrinv  = -jmass * rinv; \
      const real mrinv3 = mrinv * rinv*rinv; \
 \
      iaccx += mrinv3 * dx; \
      iaccy += mrinv3 * dy; \
      iaccz += mrinv3 * dz; \
      igpot += mrinv; \
}
    STEP(0)
    }
    accx[i]  = iaccx;
    accy[i]  = iaccy;
    accz[i]  = iaccz;
    gpotLoc += igpot;
  }
  gpotList[taskIndex] = reduce_add(gpotLoc);
#endif
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
  nTasks = nbodies/(4*programCount);
#endif
  assert((nbodies % nTasks) == 0);

  for (uniform int step = 0; step < nSteps; step++)
  { 
    launch [nTasks] updatePositions(nbodies, posx, posy, posz, velx, vely, velz,dt);
    sync;
    launch [nTasks] computeForces(nbodies, posx, posy, posz, mass);
    sync;
    launch [nTasks] updateVelocities(nbodies, posx, posy, posz, dt);
    sync;
  }

  if (energies != NULL)
  {
    real gpotLoc = 0;
    foreach (i = 0 ... nTasks)
      gpotLoc += gpotList[i];
    energies[0] = reduce_add(gpotLoc);
  }
}


