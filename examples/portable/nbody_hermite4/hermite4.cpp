/*
  Copyright (c) 2014, Evghenii Gaburov
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* Hermite4 N-body integrator */
/* Makino and Aarseth, 1992 */
/* http://adsabs.harvard.edu/abs/1992PASJ...44..141M and references there in*/

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cassert>

#include "timing.h"
#include "ispc_malloc.h"

#include "typeReal.h"
#include "hermite4_ispc.h"

struct Hermite4
{
  enum {PP_FLOP=44};
  const int n;
  const real eta;
  real eps2;
  real *g_mass, *g_gpot;
  real *g_posx, *g_posy, *g_posz;
  real *g_velx, *g_vely, *g_velz;
  real *g_accx, *g_accy, *g_accz;
  real *g_jrkx, *g_jrky, *g_jrkz;

  std::vector<real> accx0, accy0, accz0;
  std::vector<real> jrkx0, jrky0, jrkz0;

  Hermite4(const int _n = 8192, const real _eta = 0.1) : n(_n), eta(_eta)
  {
    eps2  = 4.0/n;  /* eps = 4/n to give Ebin = 1 KT */
    eps2 *= eps2;
    g_mass = new real[n];
    g_gpot = new real[n];
    g_posx = new real[n];
    g_posy = new real[n];
    g_posz = new real[n];
    g_velx = new real[n];
    g_vely = new real[n];
    g_velz = new real[n];
    g_accx = new real[n];
    g_accy = new real[n];
    g_accz = new real[n];
    g_jrkx = new real[n];
    g_jrky = new real[n];
    g_jrkz = new real[n];

    accx0.resize(n);
    accy0.resize(n);
    accz0.resize(n);
    jrkx0.resize(n);
    jrky0.resize(n);
    jrkz0.resize(n);

    printf("---Intializing nbody--- \n");

    const real R0 = 1;
    const real mp = 1.0/n;
#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++)
    {
      real xp, yp, zp, s2 = 2*R0;
      real vx, vy, vz;
      while (s2 > R0*R0) {
        xp = (1.0 - 2.0*drand48())*R0;
        yp = (1.0 - 2.0*drand48())*R0;
        zp = (1.0 - 2.0*drand48())*R0;
        s2 = xp*xp + yp*yp + zp*zp;
        vx = drand48() * 0.1;
        vy = drand48() * 0.1;
        vz = drand48() * 0.1;
      }
      g_posx[i] = xp;
      g_posy[i] = yp;
      g_posz[i] = zp;
      g_velx[i] = vx;
      g_vely[i] = vy;
      g_velz[i] = vz;
      g_mass[i] = mp;
    }
  }

  ~Hermite4()
  {
    delete g_mass;
    delete g_gpot;
    delete g_posx;
    delete g_posy;
    delete g_posz;
    delete g_velx;
    delete g_vely;
    delete g_velz;
    delete g_accx;
    delete g_accy;
    delete g_accz;
    delete g_jrkx;
    delete g_jrky;
    delete g_jrkz;
  }

  void forces();

  real step(const real dt)
  {
    const real dt2 = dt*real(1.0/2.0);
    const real dt3 = dt*real(1.0/3.0);

    real dt_min = HUGE;

#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++)
    {
      accx0[i] = g_accx[i];
      accy0[i] = g_accy[i];
      accz0[i] = g_accz[i];
      jrkx0[i] = g_jrkx[i];
      jrky0[i] = g_jrky[i];
      jrkz0[i] = g_jrkz[i];

      g_posx[i] += dt*(g_velx[i] + dt2*(g_accx[i] + dt3*g_jrkx[i]));
      g_posy[i] += dt*(g_vely[i] + dt2*(g_accy[i] + dt3*g_jrky[i]));
      g_posz[i] += dt*(g_velz[i] + dt2*(g_accz[i] + dt3*g_jrkz[i]));

      g_velx[i] += dt*(g_accx[i] + dt2*g_jrkx[i]);
      g_vely[i] += dt*(g_accy[i] + dt2*g_jrky[i]);
      g_velz[i] += dt*(g_accz[i] + dt2*g_jrkz[i]);
    }

    forces();

    if (dt > 0.0)
    {
      const real h    = dt*real(0.5);
      const real hinv = real(1.0)/h;
      const real f1   = real(0.5)*hinv*hinv;
      const real f2   = real(3.0)*hinv*f1;

      const real dt2  = dt *dt * real(1.0/2.0);
      const real dt3  = dt2*dt * real(1.0/3.0);
      const real dt4  = dt3*dt * real(1.0/4.0);
      const real dt5  = dt4*dt * real(1.0/5.0);

#pragma omp parallel for schedule(runtime) reduction(min:dt_min)
      for (int i = 0; i < n; i++)
      {
        /* compute snp & crk */

        const real Amx = g_accx[i] - accx0[i];
        const real Amy = g_accy[i] - accy0[i];
        const real Amz = g_accz[i] - accz0[i];

        const real Jmx = h*(g_jrkx[i] - jrkx0[i]);
        const real Jmy = h*(g_jrky[i] - jrky0[i]);
        const real Jmz = h*(g_jrkz[i] - jrkz0[i]);

        const real Jpx = h*(g_jrkx[i] + jrkx0[i]);
        const real Jpy = h*(g_jrky[i] + jrky0[i]);
        const real Jpz = h*(g_jrkz[i] + jrkz0[i]);


        real snpx = f1*Jmx;
        real snpy = f1*Jmy;
        real snpz = f1*Jmz;

        real crkx = f2*(Jpx - Amx);
        real crky = f2*(Jpy - Amy);
        real crkz = f2*(Jpz - Amz);

        snpx -= h*crkx;
        snpy -= h*crky;
        snpz -= h*crkz;

        /* correct */

        g_posx[i] += dt4*snpx + dt5*crkx;
        g_posy[i] += dt4*snpy + dt5*crky;
        g_posz[i] += dt4*snpz + dt5*crkz;

        g_velx[i] += dt3*snpx + dt4*crkx;
        g_vely[i] += dt3*snpy + dt4*crky;
        g_velz[i] += dt3*snpz + dt4*crkz;

        /* compute new timestep */

        const real s0 = g_accx[i]*g_accx[i] + g_accy[i]*g_accy[i] + g_accz[i]*g_accz[i];
        const real s1 = g_jrkx[i]*g_jrkx[i] + g_jrky[i]*g_jrky[i] + g_jrkz[i]*g_jrkz[i];
        const real s2 = snpx*snpx + snpy*snpy + snpz*snpz;
        const real s3 = crkx*crkx + crky*crky + crkz*crkz;

        const double u = std::sqrt(s0*s2) + s1;
        const double l = std::sqrt(s1*s3) + s2;
        assert(l > 0.0f);
        const real dt_loc = eta *std::sqrt(u/l);
        dt_min = std::min(dt_min, dt_loc);
      }
    }

    if (dt_min == HUGE)
      return dt;
    else
      return dt_min;
  }

  void energy(real &Ekin, real &Epot)
  {
    real ekin = 0, epot = 0;

#pragma omp parallel for reduction(+:ekin,epot)
    for (int i = 0; i < n; i++)
    {
      ekin += g_mass[i] * (g_velx[i]*g_velx[i] + g_vely[i]*g_vely[i] + g_velz[i]*g_velz[i]) * real(0.5f);
      epot += real(0.5f)*g_mass[i] * g_gpot[i];
    }
    Ekin = ekin;
    Epot = epot;
  }

  void integrate(const int niter, const real t_end = HUGE)
  {
    const double tin = rtc();
    forces();
    const double fn = n;
    printf(" mean flop rate in %g sec [%g GFLOP/s]\n", rtc() - tin,
        fn*fn*PP_FLOP/(rtc() - tin)/1e9);

    real Epot0, Ekin0;
    energy(Ekin0, Epot0);
    const real Etot0 = Epot0 + Ekin0;
    printf(" E: %g %g %g \n", Epot0, Ekin0, Etot0);

    /////////

    real t_global = 0;
    double t0 = 0;
    int iter = 0;
    int ntime = 10;
    real dt = 1.0/131072;
    real Epot, Ekin, Etot = Etot0;
    while (t_global < t_end) {
      if (iter % ntime == 0)
        t0 = rtc();

      if (iter >= niter) return;

      dt = step(dt);
      iter++;
      t_global += dt;

      const real Etot_pre = Etot;
      energy(Ekin, Epot);
      Etot = Ekin + Epot;

      if (iter % 1 == 0) {
        const real Etot = Ekin + Epot;
        printf("iter= %d: t= %g  dt= %g Ekin= %g  Epot= %g  Etot= %g , dE = %g d(dE)= %g \n",
            iter, t_global, dt, Ekin, Epot, Etot, (Etot - Etot0)/std::abs(Etot0),
            (Etot - Etot_pre)/std::abs(Etot_pre)   );
      }

      if (iter % ntime == 0) {
        printf(" mean flop rate in %g sec [%g GFLOP/s]\n", rtc() - t0,
            fn*fn*PP_FLOP/(rtc() - t0)/1e9*ntime);
      }

      fflush(stdout);

    }
  }

};



void Hermite4::forces()
{
  ispc::compute_forces(
      n,
      g_mass,
      g_posx,
      g_posy,
      g_posz,
      g_velx,
      g_vely,
      g_velz,
      g_accx,
      g_accy,
      g_accz,
      g_jrkx,
      g_jrky,
      g_jrkz,
      g_gpot,
      eps2);
}

void run(const int nbodies, const real eta, const int nstep)
{
  Hermite4 h4(nbodies, eta);
  h4.integrate(nstep);
}

int main(int argc, char *argv[])
{
  printf("  Usage: %s [nbodies=8192] [nsteps=40] [eta=0.1] \n", argv[0]);

  int nbodies = 8192;
  if (argc > 1) nbodies = atoi(argv[1]);

  int nstep = 40;
  if (argc > 2) nstep = atoi(argv[2]);

  float eta = 0.1;
  if (argc > 3) eta = atof(argv[3]);



  printf("nbodies= %d\n", nbodies);
  printf("nstep= %d\n", nstep);
  printf(" eta= %g \n", eta);

  run(nbodies, eta, nstep);

  return 0;
}

