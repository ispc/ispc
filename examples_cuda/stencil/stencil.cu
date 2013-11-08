#define programCount 32
#define programIndex (threadIdx.x & 31)
#define taskIndex (blockIdx.x*4 + (threadIdx.x >> 5))

__device__ static void
stencil_step( int x0,  int x1,
              int y0,  int y1,
              int z0,  int z1,
              int Nx,  int Ny,  int Nz,
              const double coef[4],  const double vsq[],
              const double Ain[],  double Aout[]) {
    const  int Nxy = Nx * Ny;


#if 0
    foreach (z = z0 ... z1, y = y0 ... y1, x = x0 ... x1) {
#else
      const  double coef0 = coef[0];
      const  double coef1 = coef[1];
      const  double coef2 = coef[2];
      const  double coef3 = coef[3];
      for ( int z = z0; z < z1; z++)
        for ( int y = y0 ; y < y1; y++)
          for ( int xb = x0; xb < x1; xb += programCount)
          {
            const int x = xb + programIndex;

#endif
        int index = (z * Nxy) + (y * Nx) + x;
#define A_cur(x, y, z) Ain[index + (x) + ((y) * Nx) + ((z) * Nxy)]
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
            vsq[index] * div;
    }
}


extern "C"
__global__  void
stencil_step_task( int x0,  int x1,
                   int y0,  int y1,
                   int z0,
                   int Nx,  int Ny,  int Nz,
                   const double coef[4],  const double vsq[],
                   const double Ain[],  double Aout[]) {
    stencil_step(x0, x1, y0, y1, z0+taskIndex, z0+taskIndex+1,
                 Nx, Ny, Nz, coef, vsq, Ain, Aout);
}

