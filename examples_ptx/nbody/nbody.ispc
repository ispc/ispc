typedef double real;

typedef real<3> real3;
typedef real<4> real4;

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
real4 ppForce(real3 ipos, real3 jpos, real jmass)
{
  const real3   dr  = jpos - ipos;
  const real    r2  = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  const real  rinv  = r2 > 0 ? rsqrt(r2) : 0;
  const real mrinv  = jmass * rinv;
  const real mrinv3 = mrinv * rinv*rinv;

  real4 acc;
  acc.x = -mrinv3 * dr.x;
  acc.y = -mrinv3 * dr.y;
  acc.z = -mrinv3 * dr.z;
  acc.w = -mrinv;

  return acc;
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
  real gpotLoc = 0;
  for (uniform int i = blockBeg; i < blockEnd; i++)
  {
    const real3 ipos = {posx[i], posy[i], posz[i]};
    real4 iacc = 0;
    foreach (j = 0 ... nbodies)
    {
      const real3 jpos = {posx[j], posy[j], posz[j]};
      const real jmass = mass[j];
      iacc += ppForce(ipos, jpos, jmass);
    }
    accx[i]  = reduce_add(iacc.x);
    accy[i]  = reduce_add(iacc.y);
    accz[i]  = reduce_add(iacc.z);
    gpotLoc += reduce_add(iacc.w);
  }
  atomic_add_global(&gpot, gpotLoc);
#else
  real gpotLoc = 0;
  foreach (i = blockBeg ... blockEnd)
  {
    const real3 ipos = {posx[i], posy[i], posz[i]};
    real4 iacc = 0;
    for (uniform int j = 0; j < nbodies; j++)
    {
      const real3 jpos = {posx[j], posy[j], posz[j]};
      const real jmass = mass[j];
      iacc += ppForce(ipos, jpos, jmass);
    }
    accx[i]  = iacc.x;
    accy[i]  = iacc.y;
    accz[i]  = iacc.z;
    gpotLoc += iacc.w;
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


