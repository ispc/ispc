#include "cuda_helpers.cuh"

__device__ static void
stencil_step( int x0,  int x1,
              int y0,  int y1,
              int z0,  int z1,
              int Nx,  int Ny,  int Nz,
              const double coef[4],  const double vsq[],
              const double Ain[],  double Aout[]) 
{
  const  int Nxy = Nx * Ny;


  const  double coef0 = coef[0];
  const  double coef1 = coef[1];
  const  double coef2 = coef[2];
  const  double coef3 = coef[3];
  for ( int z = z0; z < z1; z++)
    for ( int y = y0 ; y < y1; y++)
      for ( int xb = x0; xb < x1; xb += programCount)
      {
        const int x = xb + programIndex;

        int index = (z * Nxy) + (y * Nx) + x;
#define A_cur(x, y, z) __ldg(&Ain[index + (x) + ((y) * Nx) + ((z) * Nxy)])
#define A_next(x, y, z) Aout[index + (x) + ((y) * Nx) + ((z) * Nxy)]
        double div = 
          coef0 *  A_cur(0, 0, 0) +
          coef1 * (A_cur(+1, 0, 0) + A_cur(-1, 0, 0) +
              A_cur(0, +1, 0) + A_cur(0, -1, 0) +
              A_cur(0, 0, +1) + A_cur(0, 0, -1)) +
          coef2 * (A_cur(+2, 0, 0) + A_cur(-2, 0, 0) +
              A_cur(0, +2, 0) + A_cur(0, -2, 0) +
              A_cur(0, 0, +2) + A_cur(0, 0, -2)) +
          coef3 * (A_cur(+3, 0, 0) + A_cur(-3, 0, 0) +
              A_cur(0, +3, 0) + A_cur(0, -3, 0) +
              A_cur(0, 0, +3) + A_cur(0, 0, -3));

        if (x < x1)
          A_next(0, 0, 0) = 2.0 * A_cur(0, 0, 0) - A_next(0, 0, 0) + 
            __ldg(&vsq[index]) * div;
      }
}


#define SPANX 32
#define SPANY 2
#define SPANZ 4

__global__  void
stencil_step_task( int x0,  int x1,
                   int y0,  int y1,
                   int z0,  int z1,
                   int Nx,  int Ny,  int Nz,
                   const double coef[4],  const double vsq[],
                   const double Ain[],  double Aout[]) {
  if (taskIndex0 >= taskCount0 || 
      taskIndex1 >= taskCount1 || 
      taskIndex2 >= taskCount2)
    return;

  const  int xfirst = x0 + taskIndex0 * SPANX;
  const  int xlast  = min(x1, xfirst + SPANX);

  const  int yfirst = y0 + taskIndex1 * SPANY;
  const  int ylast  = min(y1, yfirst + SPANY);

  const  int zfirst = z0 + taskIndex2 * SPANZ;
  const  int zlast  = min(z1, zfirst + SPANZ);

  stencil_step(xfirst,xlast, yfirst,ylast, zfirst,zlast,
      Nx, Ny, Nz, coef, vsq, Ain, Aout);
}



extern "C"
__global__ void
loop_stencil_ispc_tasks___export( int t0,  int t1, 
                         int x0,  int x1,
                         int y0,  int y1,
                         int z0,  int z1,
                         int Nx,  int Ny,  int Nz,
                         const double coef[4], 
                         const double vsq[],
                         double Aeven[],  double Aodd[])
{
#define NB(x,n) (((x)+(n)-1)/(n))

  dim3 grid((NB(x1-x0,SPANX)-1)/4+1, NB(y1-y0,SPANY), NB(z1-z0,SPANZ));

    for ( int t = t0; t < t1; ++t) 
    {
      // Parallelize across cores as well: each task will work on a slice
      // of 1 in the z extent of the volume.
      if ((t & 1) == 0)
      {
        if (programIndex == 0)
          stencil_step_task<<<grid,128>>>(x0, x1, y0, y1, z0, z1, Nx, Ny, Nz, 
              coef, vsq, Aeven, Aodd);
      }
      else
      {
        if (programIndex == 0)
          stencil_step_task<<<grid,128>>>(x0, x1, y0, y1, z0, z1, Nx, Ny, Nz, 
              coef, vsq, Aodd, Aeven);
      }

      // We need to wait for all of the launched tasks to finish before
      // starting the next iteration
      cudaDeviceSynchronize();
    }
}

extern "C"
__host__ void
loop_stencil_ispc_tasks( int t0,  int t1, 
                         int x0,  int x1,
                         int y0,  int y1,
                         int z0,  int z1,
                         int Nx,  int Ny,  int Nz,
                         const double coef[4], 
                         const double vsq[],
                         double Aeven[],  double Aodd[])
{
  loop_stencil_ispc_tasks___export<<<1,32>>>(t0,t1,x0,x1,y0,y1,z0,z1,Nx,Ny,Nz,coef,vsq,Aeven,Aodd);
  cudaDeviceSynchronize();
}

