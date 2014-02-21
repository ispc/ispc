#include "typeReal.h"

typedef real<3> vec3;
struct Force
{
  vec3 acc, jrk;
  real pot, null;
};

struct Predictor
{
  vec3 pos, vel;
};

static inline 
void body_body_force(
    Force &fi,
    const Predictor &pi,
    const Predictor &pj,
    const real mj)
{
  const real dx = pj.pos.x - pi.pos.x;
  const real dy = pj.pos.y - pi.pos.y;
  const real dz = pj.pos.z - pi.pos.z;

  const real ds2 = dx*dx + dy*dy + dz*dz;

  const real  inv_ds  = ds2 > 1.0d-10 ? rsqrt((float)ds2) : 0.0;
  const real  inv_ds2 = inv_ds*inv_ds;
  const real minv_ds  = inv_ds  * mj;
  const real minv_ds3 = inv_ds2 * minv_ds;


  fi.acc.x += minv_ds3 * dx;
  fi.acc.y += minv_ds3 * dy;
  fi.acc.z += minv_ds3 * dz;
  fi.pot   -= minv_ds;
  
  const real dvx = pj.vel.x - pi.vel.x;
  const real dvy = pj.vel.y - pi.vel.y;
  const real dvz = pj.vel.z - pi.vel.z;
  const real rv  = dx*dvx + dy*dvy + dz*dvz;
  
  const real Jij = (real)(-3.0) * (rv * inv_ds2 * minv_ds3);
  
  fi.jrk.x += minv_ds3*dvx + Jij*dx;
  fi.jrk.y += minv_ds3*dvy + Jij*dy;
  fi.jrk.z += minv_ds3*dvz + Jij*dz;
}

task void compute_forces_task(
    uniform const int     n,
    uniform const int nPerTask,
    uniform const real mass[],
    uniform const real posx[],
    uniform const real posy[],
    uniform const real posz[],
    uniform const real velx[],
    uniform const real vely[],
    uniform const real velz[],
    uniform       real accx[],
    uniform       real accy[],
    uniform       real accz[],
    uniform       real jrkx[],
    uniform       real jrky[],
    uniform       real jrkz[],
    uniform       real gpot[])
{
  const uniform int nibeg = taskIndex * nPerTask;
  const uniform int niend = min(n, nibeg + nPerTask);

  if (nibeg >= n) 
    return;

  uniform real shdata[7][programCount];

  assert((n%programCount) == 0);

  foreach (i = nibeg ... niend)
  {
    Force fi;
    fi.acc = (real)0.0;
    fi.jrk = (real)0.0;
    fi.pot = (real)0.0;
   
    Predictor pi;
    pi.pos.x = posx[i];
    pi.pos.y = posy[i];
    pi.pos.z = posz[i];
    pi.vel.x = velx[i];
    pi.vel.y = vely[i];
    pi.vel.z = velz[i];

    for (uniform int jb = 0; jb < n; jb += programCount)
    {
      const int jp = jb + programIndex;
      shdata[0][programIndex] = posx[jp];
      shdata[1][programIndex] = posy[jp];
      shdata[2][programIndex] = posz[jp];
      shdata[3][programIndex] = mass[jp];
      shdata[4][programIndex] = velx[jp];
      shdata[5][programIndex] = vely[jp];
      shdata[6][programIndex] = velz[jp];

      for (uniform int j = 0; j < programCount; j++)
      {
        Predictor pj;
        pj.pos.x = shdata[0][j];
        pj.pos.y = shdata[1][j];
        pj.pos.z = shdata[2][j];
        pj.vel.x = shdata[4][j];
        pj.vel.y = shdata[5][j];
        pj.vel.z = shdata[6][j];
        const real jmass  = shdata[3][j];
        body_body_force(fi,pi,pj,jmass);
      }
    }

    accx[i] = fi.acc.x;
    accy[i] = fi.acc.y;
    accz[i] = fi.acc.z;
    jrkx[i] = fi.jrk.x;
    jrky[i] = fi.jrk.y;
    jrkz[i] = fi.jrk.z;
    gpot[i] = fi.pot;
  }
}

export void compute_forces(
    uniform const int     n,
    uniform const real mass[],
    uniform const real posx[],
    uniform const real posy[],
    uniform const real posz[],
    uniform const real velx[],
    uniform const real vely[],
    uniform const real velz[],
    uniform       real accx[],
    uniform       real accy[],
    uniform       real accz[],
    uniform       real jrkx[],
    uniform       real jrky[],
    uniform       real jrkz[],
    uniform       real gpot[])
{
  const uniform int nPerTask = min(128,programCount*8);
  const uniform int nTask = (n+nPerTask-1)/nPerTask;

  launch [nTask]  compute_forces_task(
      n, nPerTask,
      mass, 
      posx,posy,posz,
      velx,vely,velz,
      accx,accy,accz,
      jrkx,jrky,jrkz,
      gpot);
}
