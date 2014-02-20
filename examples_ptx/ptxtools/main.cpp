/*

Bonsai V2: A parallel GPU N-body gravitational Tree-code

(c) 2010-2012:
Jeroen Bedorf
Evghenii Gaburov
Simon Portegies Zwart

Leiden Observatory, Leiden University

http://castle.strw.leidenuniv.nl
http://github.com/treecode/Bonsai

*/

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <process.h>
#define M_PI        3.14159265358979323846264338328

#include <stdlib.h>
#include <time.h>
void srand48(const long seed)
{
  srand(seed);
}
//JB This is not a proper work around but just to get things compiled...
double drand48()
{
  return double(rand())/RAND_MAX;
}


#endif


#ifdef USE_MPI
  #include <omp.h>
  #include <mpi.h>
#endif

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <sstream>
#include "log.h"
#include "anyoption.h"
#include "renderloop.h"
#include "plummer.h"
#include "disk_shuffle.h"
#ifdef GALACTICS
#include "galactics.h"
#endif


#if ENABLE_LOG
  bool ENABLE_RUNTIME_LOG;
  bool PREPEND_RANK;
  int  PREPEND_RANK_PROCID;
  int  PREPEND_RANK_NPROCS;
#endif

using namespace std;

#include "../profiling/bonsai_timing.h"

int devID;
int renderDevID;

extern void initTimers()
{
#ifndef CUXTIMER_DISABLE
  // Set up the profiling timing info
  build_tree_init();
  compute_propertiesD_init();
  dev_approximate_gravity_init();
  parallel_init();
  sortKernels_init();
  timestep_init();
#endif
}

extern void displayTimers()
{
#ifndef CUXTIMER_DISABLE
  // Display all timing info on the way out
  build_tree_display();
  compute_propertiesD_display();
  //dev_approximate_gravity_display();
  //parallel_display();
  //sortKernels_display();
  //timestep_display();
#endif
}

#include "octree.h"

#ifdef USE_OPENGL
#include "renderloop.h"
#include <cuda_gl_interop.h>
#endif

void read_dumbp_file_parallel(vector<real4> &bodyPositions, vector<real4> &bodyVelocities,  vector<int> &bodiesIDs,  float eps2,
                     string fileName, int rank, int procs, int &NTotal2, int &NFirst, int &NSecond, int &NThird, octree *tree, int reduce_bodies_factor)  
{
  //Process 0 does the file reading and sends the data
  //to the other processes
  
  //Now we have different types of files, try to determine which one is used
  /*****
  If individual softening is on there is only one option:
  Header is formatted as follows: 
  N     #       #       #
  so read the first number and compute how particles should be distributed
  
  If individual softening is NOT enabled, i can be anything, but for ease I assume standard dumbp files:
  no Header
  ID mass x y z vx vy vz
  now the next step is risky, we assume mass adds up to 1, so number of particles will be : 1 / mass
  use this as initial particle distribution
  
  */
  
  
  char fullFileName[256];
  sprintf(fullFileName, "%s", fileName.c_str());

  LOG("Trying to read file: %s \n", fullFileName);

  ifstream inputFile(fullFileName, ios::in);

  if(!inputFile.is_open())
  {
    LOG("Can't open input file \n");
    exit(0);
  }
  
  int NTotal;
  int idummy;
  real4 positions;
  real4 velocity;

  #ifndef INDSOFT
     inputFile >> idummy >> positions.w;
     inputFile.seekg(0, ios::beg); //Reset file pointer
     NTotal = (int)(1 / positions.w);
  #else
     //Read the Ntotal from the file header
     inputFile >> NTotal >> NFirst >> NSecond >> NThird;
  #endif
  
  
  
  //Rough divide
  uint perProc = NTotal / procs;
  bodyPositions.reserve(perProc+10);
  bodyVelocities.reserve(perProc+10);
  bodiesIDs.reserve(perProc+10);
  perProc -= 1;

  //Start reading
  int particleCount = 0;
  int procCntr = 1;

  int globalParticleCount = 0;

  while(!inputFile.eof()) {
    
    inputFile >> idummy
              >> positions.w >> positions.x >> positions.y >> positions.z
              >> velocity.x >> velocity.y >> velocity.z;    

	globalParticleCount++;

	if( globalParticleCount % reduce_bodies_factor == 0 ) 
		positions.w *= reduce_bodies_factor;

	if( globalParticleCount % reduce_bodies_factor != 0 )
		continue;

    #ifndef INDSOFT
      velocity.w = sqrt(eps2);
    #else
      inputFile >> velocity.w; //Read the softening from the input file
    #endif
    
    bodyPositions.push_back(positions);
    bodyVelocities.push_back(velocity);
    
    #ifndef INDSOFT    
      idummy = particleCount;
    #endif
    
    bodiesIDs.push_back(idummy);  
    
    particleCount++;
  
  
    if(bodyPositions.size() > perProc && procCntr != procs)
    {       
      tree->ICSend(procCntr,  &bodyPositions[0], &bodyVelocities[0],  &bodiesIDs[0], (int)bodyPositions.size());
      procCntr++;
      
      bodyPositions.clear();
      bodyVelocities.clear();
      bodiesIDs.clear();
    }
  }//end while
  
  inputFile.close();
  
  //Clear the last one since its double
  bodyPositions.resize(bodyPositions.size()-1);  
  NTotal2 = particleCount-1;
  
  LOGF(stderr, "NTotal:  %d\tper proc: %d\tFor ourself: %d \n", NTotal, perProc, (int)bodiesIDs.size());
}

void read_tipsy_file_parallel(vector<real4> &bodyPositions, vector<real4> &bodyVelocities,
                              vector<int> &bodiesIDs,  float eps2, string fileName, 
                              int rank, int procs, int &NTotal2, int &NFirst, 
                              int &NSecond, int &NThird, octree *tree,
                              vector<real4> &dustPositions, vector<real4> &dustVelocities,
                              vector<int> &dustIDs, int reduce_bodies_factor,
                              int reduce_dust_factor)  
{
  //Process 0 does the file reading and sends the data
  //to the other processes
  /* 

     Read in our custom version of the tipsy file format.
     Most important change is that we store particle id on the 
     location where previously the potential was stored.
  */
  
  
  char fullFileName[256];
  sprintf(fullFileName, "%s", fileName.c_str());

  LOG("Trying to read file: %s \n", fullFileName);
  
  
  
  ifstream inputFile(fullFileName, ios::in | ios::binary);
  if(!inputFile.is_open())
  {
    LOG("Can't open input file \n");
    exit(0);
  }
  
  dump  h;
  inputFile.read((char*)&h, sizeof(h));  

  int NTotal;
  int idummy;
  real4 positions;
  real4 velocity;

     
  //Read tipsy header  
  NTotal        = h.nbodies;
  NFirst        = h.ndark;
  NSecond       = h.nstar;
  NThird        = h.nsph;

  tree->set_t_current((float) h.time);
  
  //Rough divide
  uint perProc = (NTotal / procs) /reduce_bodies_factor;
  bodyPositions.reserve(perProc+10);
  bodyVelocities.reserve(perProc+10);
  bodiesIDs.reserve(perProc+10);
  perProc -= 1;

  //Start reading
  int particleCount = 0;
  int procCntr = 1;
  
  dark_particle d;
  star_particle s;

  int globalParticleCount = 0;
  int bodyCount = 0;
  int dustCount = 0;
  
  for(int i=0; i < NTotal; i++)
  {
    if(i < NFirst)
    {
      inputFile.read((char*)&d, sizeof(d));
      velocity.w        = d.eps;
      positions.w       = d.mass;
      positions.x       = d.pos[0];
      positions.y       = d.pos[1];
      positions.z       = d.pos[2];
      velocity.x        = d.vel[0];
      velocity.y        = d.vel[1];
      velocity.z        = d.vel[2];
      idummy            = d.phi;
    }
    else
    {
      inputFile.read((char*)&s, sizeof(s));
      velocity.w        = s.eps;
      positions.w       = s.mass;
      positions.x       = s.pos[0];
      positions.y       = s.pos[1];
      positions.z       = s.pos[2];
      velocity.x        = s.vel[0];
      velocity.y        = s.vel[1];
      velocity.z        = s.vel[2];
      idummy            = s.phi;
    }


    if(positions.z < -10e10)
    {
       fprintf(stderr," Removing particle %d because of Z is: %f \n", globalParticleCount, positions.z);
       continue;
    }


	globalParticleCount++;
   
    #ifdef USE_DUST
      if(idummy >= 50000000 && idummy < 100000000)
      {
        dustCount++;
        if( dustCount % reduce_dust_factor == 0 ) 
          positions.w *= reduce_dust_factor;

        if( dustCount % reduce_dust_factor != 0 )
          continue;
        dustPositions.push_back(positions);
        dustVelocities.push_back(velocity);
        dustIDs.push_back(idummy);      
      }
      else
      {
        bodyCount++;
        if( bodyCount % reduce_bodies_factor == 0 ) 
		      positions.w *= reduce_bodies_factor;

	      if( bodyCount % reduce_bodies_factor != 0 )
		      continue;
        bodyPositions.push_back(positions);
        bodyVelocities.push_back(velocity);
        bodiesIDs.push_back(idummy);  
      }

    
    #else
      if( globalParticleCount % reduce_bodies_factor == 0 ) 
        positions.w *= reduce_bodies_factor;

      if( globalParticleCount % reduce_bodies_factor != 0 )
        continue;
      bodyPositions.push_back(positions);
      bodyVelocities.push_back(velocity);
      bodiesIDs.push_back(idummy);  
    #endif
    
    particleCount++;
  
  
    if(bodyPositions.size() > perProc && procCntr != procs)
    { 
      tree->ICSend(procCntr,  &bodyPositions[0], &bodyVelocities[0],  &bodiesIDs[0], (int)bodyPositions.size());
      procCntr++;
      
      bodyPositions.clear();
      bodyVelocities.clear();
      bodiesIDs.clear();
    }
  }//end while
  
  inputFile.close();
  
  //Clear the last one since its double
//   bodyPositions.resize(bodyPositions.size()-1);  
//   NTotal2 = particleCount-1;
  NTotal2 = particleCount;
  LOGF(stderr,"NTotal: %d\tper proc: %d\tFor ourself: %d \tNDust: %d \n",
               NTotal, perProc, (int)bodiesIDs.size(), (int)dustPositions.size());
}


void read_generate_cube(vector<real4> &bodyPositions, vector<real4> &bodyVelocities,
                              vector<int> &bodiesIDs,  float eps2, string fileName, 
                              int rank, int procs, int &NTotal2, int &NFirst, 
                              int &NSecond, int &NThird, octree *tree,
                              vector<real4> &dustPositions, vector<real4> &dustVelocities,
                              vector<int> &dustIDs, int reduce_bodies_factor,
                              int reduce_dust_factor)  
{
  //Process 0 does the file reading and sends the data
  //to the other processes
  /* 

     Read in our custom version of the tipsy file format.
     Most important change is that we store particle id on the 
     location where previously the potential was stored.
  */
  

  int NTotal;
  int idummy;
  real4 positions;
  real4 velocity;

     
  //Read tipsy header  
  NTotal        = (int)std::pow(2.0, 22);
  NFirst        = NTotal;
  NSecond       = 0;
  NThird        = 0;

  fprintf(stderr,"Going to generate a random cube , number of particles: %d \n", NTotal);

  tree->set_t_current((float) 0);
  
  //Rough divide
  uint perProc = NTotal / procs;
  bodyPositions.reserve(perProc+10);
  bodyVelocities.reserve(perProc+10);
  bodiesIDs.reserve(perProc+10);
  perProc -= 1;

  //Start reading
  int particleCount = 0;
  int procCntr = 1;
  

  int globalParticleCount = 0;

  float mass = 1.0  / NTotal;
  
  for(int i=0; i < NTotal; i++)
  {
      velocity.w        = 0;
      positions.w       = mass;
      positions.x       = drand48();
      positions.y       = drand48();
      positions.z       = drand48();
      velocity.x        = 0.001*drand48();
      velocity.y        = 0.001*drand48();
      velocity.z        = 0.001*drand48();

      globalParticleCount++;
      bodyPositions.push_back(positions);
      bodyVelocities.push_back(velocity);
      bodiesIDs.push_back(globalParticleCount);  
  
    if(bodyPositions.size() > perProc && procCntr != procs)
    { 
      tree->ICSend(procCntr,  &bodyPositions[0], &bodyVelocities[0],  &bodiesIDs[0], (int)bodyPositions.size());
      procCntr++;
      
      bodyPositions.clear();
      bodyVelocities.clear();
      bodiesIDs.clear();
    }
  }//end while
  
  //Clear the last one since its double
//   bodyPositions.resize(bodyPositions.size()-1);  
//   NTotal2 = particleCount-1;
  NTotal2 = NTotal;
  LOGF(stderr,"NTotal: %d\tper proc: %d\tFor ourself: %d \tNDust: %d \n",
               NTotal, perProc, (int)bodiesIDs.size(), (int)dustPositions.size());
}


double rot[3][3];

void rotmat(double i,double w)
{
    rot[0][0] = cos(w);
    rot[0][1] = -cos(i)*sin(w);
    rot[0][2] = -sin(i)*sin(w);
    rot[1][0] = sin(w);
    rot[1][1] = cos(i)*cos(w);
    rot[1][2] = sin(i)*cos(w);
    rot[2][0] = 0.0;
    rot[2][1] = -sin(i);
    rot[2][2] = cos(i);
    fprintf(stderr,"%g %g %g\n",rot[0][0], rot[0][1], rot[0][2]);
    fprintf(stderr,"%g %g %g\n",rot[1][0], rot[1][1], rot[1][2]);
    fprintf(stderr,"%g %g %g\n",rot[2][0], rot[2][1], rot[2][2]);
}

void rotate(double rot[3][3],float *vin)
{
    static double vout[3];

    for(int i=0; i<3; i++) {
      vout[i] = 0;
      for(int j=0; j<3; j++)
        vout[i] += rot[i][j] * vin[j]; 
      /* Remember the rotation matrix is the transpose of rot */
    }
    for(int i=0; i<3; i++)
            vin[i] = (float) vout[i];
}

void euler(vector<real4> &bodyPositions,
           vector<real4> &bodyVelocities,
           double inc, double omega)
{
  rotmat(inc,omega);
  size_t nobj = bodyPositions.size();
  for(uint i=0; i < nobj; i++)
  {
      float r[3], v[3];
      r[0] = bodyPositions[i].x;
      r[1] = bodyPositions[i].y;
      r[2] = bodyPositions[i].z;
      v[0] = bodyVelocities[i].x;
      v[1] = bodyVelocities[i].y;
      v[2] = bodyVelocities[i].z;

      rotate(rot,r);
      rotate(rot,v);

      bodyPositions[i].x = r[0]; 
      bodyPositions[i].y = r[1]; 
      bodyPositions[i].z = r[2]; 
      bodyVelocities[i].x = v[0];
      bodyVelocities[i].y = v[1];
      bodyVelocities[i].z = v[2];
  }
}



double centerGalaxy(vector<real4> &bodyPositions,
                    vector<real4> &bodyVelocities)
{
    size_t nobj = bodyPositions.size();
    float xc, yc, zc, vxc, vyc, vzc, mtot;
  

    mtot = 0;
    xc = yc = zc = vxc = vyc = vzc = 0;
    for(uint i=0; i< nobj; i++) {
            xc   += bodyPositions[i].w*bodyPositions[i].x;
            yc   += bodyPositions[i].w*bodyPositions[i].y;
            zc   += bodyPositions[i].w*bodyPositions[i].z;
            vxc  += bodyPositions[i].w*bodyVelocities[i].x;
            vyc  += bodyPositions[i].w*bodyVelocities[i].y;
            vzc  += bodyPositions[i].w*bodyVelocities[i].z;
            mtot += bodyPositions[i].w;
    }
    xc /= mtot;
    yc /= mtot;
    zc /= mtot;
    vxc /= mtot;
    vyc /= mtot;
    vzc /= mtot;
    for(uint i=0; i< nobj; i++)
    {
      bodyPositions[i].x  -= xc;
      bodyPositions[i].y  -= yc;
      bodyPositions[i].z  -= zc;
      bodyVelocities[i].x -= vxc;
      bodyVelocities[i].y -= vyc;
      bodyVelocities[i].z -= vzc;
    }
    
    return mtot;
}




int setupMergerModel(vector<real4> &bodyPositions1,
                     vector<real4> &bodyVelocities1,
                     vector<int>   &bodyIDs1,
                     vector<real4> &bodyPositions2,
                     vector<real4> &bodyVelocities2,
                     vector<int>   &bodyIDs2){
        uint i;
        double ds=1.0, vs, ms=1.0;
        double mu1, mu2, vp;
        double b=1.0, rsep=10.0;
        double x, y, vx, vy, x1, y1, vx1, vy1 ,  x2, y2, vx2, vy2;
        double theta, tcoll;
        double inc1=0, omega1=0;
        double inc2=0, omega2=0;
        
        
        ds = 1.52;
        ms = 1.0;
        b = 10;
        rsep = 168;
        inc1 = 0;
        omega1 = 0;
        inc2 = 180;
        omega2 = 0;


        if(ds < 0)
        {
          cout << "Enter size ratio (for gal2): ";
          cin >> ds;
          cout << "Enter mass ratio (for gal2): ";
          cin >> ms;
          cout << "Enter relative impact parameter: ";
          cin >> b;
          
          cout << "Enter initial separation: ";
          cin >> rsep;
          cout << "Enter Euler angles for first galaxy:\n";
          cout << "Enter inclination: ";
          cin >> inc1;
          cout << "Enter omega: ";
          cin >> omega1;
          cout << "Enter Euler angles for second galaxy:\n";
          cout << "Enter inclination: ";
          cin >> inc2;
          cout << "Enter omega: ";
          cin >> omega2;
        }


        double inc1_inp, inc2_inp, om2_inp, om1_inp;
        
        inc1_inp = inc1;
        inc2_inp = inc2;
        om1_inp = omega1;
        om2_inp = omega1;


        inc1   *= M_PI/180.;
        inc2   *= M_PI/180.;
        omega1 *= M_PI/180.;
        omega2 *= M_PI/180.;
        omega1 += M_PI;

        fprintf(stderr,"Size ratio: %f Mass ratio: %f \n", ds, ms);
        fprintf(stderr,"Relative impact par: %f Initial sep: %f \n", b, rsep);
        fprintf(stderr,"Euler angles first: %f %f Second: %f %f \n",
                        inc1, omega1,inc2,omega2);

        vs = sqrt(ms/ds); /* adjustment for internal velocities */


        //Center everything in galaxy 1 and galaxy 2
        double galaxyMass1 = centerGalaxy(bodyPositions1, bodyVelocities1);
        double galaxyMass2 = centerGalaxy(bodyPositions2, bodyVelocities2);


        galaxyMass2 = ms*galaxyMass2;             //Adjust total mass

        mu1 =  galaxyMass2/(galaxyMass1 + galaxyMass2);
        mu2 = -galaxyMass1/(galaxyMass1 + galaxyMass2);
        
        double m1 = galaxyMass1;
        double m2 = galaxyMass2;

        
        /* Relative Parabolic orbit - anti-clockwise */
        if( b > 0 ) {
                vp = sqrt(2.0*(m1 + m2)/b);
                x = 2*b - rsep;  y = -2*sqrt(b*(rsep-b));
                vx = sqrt(b*(rsep-b))*vp/rsep; vy = b*vp/rsep;
        }
        else {
                b = 0;
                x = - rsep; y = 0.0;
                vx = sqrt(2.0*(m1 + m2)/rsep); vy = 0.0;
        }

        /* Calculate collison time */
        if( b > 0 ) {
                theta = atan2(y,x);
                tcoll = (0.5*tan(0.5*theta) + pow(tan(0.5*theta),3.0)/6.)*4*b/vp;
                fprintf(stderr,"Collision time is t=%g\n",tcoll);
        }
        else {
                tcoll = -pow(rsep,1.5)/(1.5*sqrt(2.0*(m1+m2)));
                fprintf(stderr,"Collision time is t=%g\n",tcoll);
        }

        /* These are the orbital adjustments for a parabolic encounter */
        /* Change to centre of mass frame */
        x1  =  mu1*x;  x2   =  mu2*x;     
        y1  =  mu1*y;  y2   =  mu2*y;
        vx1 =  mu1*vx; vx2  =  mu2*vx;
        vy1 =  mu1*vy; vy2  =  mu2*vy;


        /* Rotate the galaxies */
        euler(bodyPositions1, bodyVelocities1, inc1,omega1);
        euler(bodyPositions2, bodyVelocities2, inc2,omega2);

        for(i=0; i< bodyPositions1.size(); i++) {
                bodyPositions1[i].x  = (float) (bodyPositions1[i].x  + x1);
                bodyPositions1[i].y  = (float) (bodyPositions1[i].y  + y1);
                bodyVelocities1[i].x = (float) (bodyVelocities1[i].x + vx1);
                bodyVelocities1[i].y = (float) (bodyVelocities1[i].y + vy1);
        }
        /* Rescale and reset the second galaxy */
        for(i=0; i< bodyPositions2.size(); i++) {
                bodyPositions2[i].w = (float) ms*bodyPositions2[i].w;
                bodyPositions2[i].x = (float) (ds*bodyPositions2[i].x + x2);
                bodyPositions2[i].y = (float) (ds*bodyPositions2[i].y + y2);
                bodyPositions2[i].z = (float) ds*bodyPositions2[i].z;
                bodyVelocities2[i].x = (float) (vs*bodyVelocities2[i].x + vx2);
                bodyVelocities2[i].y = (float) (vs*bodyVelocities2[i].y + vy2);
                bodyVelocities2[i].z = (float) vs*bodyVelocities2[i].z;
        }


        //Put them into one 
        bodyPositions1.insert(bodyPositions1.end(),  bodyPositions2.begin(), bodyPositions2.end());
        bodyVelocities1.insert(bodyVelocities1.end(), bodyVelocities2.begin(), bodyVelocities2.end());
        bodyIDs1.insert(bodyIDs1.end(), bodyIDs2.begin(), bodyIDs2.end());
  

        return 0;
}


long long my_dev::base_mem::currentMemUsage;
long long my_dev::base_mem::maxMemUsage;

int main(int argc, char** argv)
{
  my_dev::base_mem::currentMemUsage = 0;
  my_dev::base_mem::maxMemUsage     = 0;

  vector<real4> bodyPositions;
  vector<real4> bodyVelocities;
  vector<int>   bodyIDs;

  vector<real4> dustPositions;
  vector<real4> dustVelocities;
  vector<int>   dustIDs;  
  

  float eps      = 0.05f;
  float theta    = 0.75f;
  float timeStep = 1.0f / 16.0f;
  float tEnd     = 1;
  int   iterEnd  = (1 << 30);
  devID      = 0;
  renderDevID = 0;

  string fileName       =  "";
  string logFileName    = "gpuLog.log";
  string snapshotFile   = "snapshot_";
  float snapshotIter     = -1;
  float  remoDistance   = -1.0;
  int    snapShotAdd    =  0;
  int rebuild_tree_rate = 2;
  int reduce_bodies_factor = 1;
  int reduce_dust_factor = 1;
  string fullScreenMode = "";
  bool direct = false;
  bool fullscreen = false;
  bool displayFPS = false;
  bool diskmode = false;
  bool stereo   = false;

#if ENABLE_LOG
  ENABLE_RUNTIME_LOG = false;
  PREPEND_RANK       = false;
#endif

#ifdef USE_OPENGL
	TstartGlow = 0.0;
	dTstartGlow = 1.0;
#endif

  int nPlummer  = -1;
  int nSphere   = -1;
  int nMilkyWay = -1;
  int nMWfork   =  4;
	/************** beg - command line arguments ********/
#if 1
	{
		AnyOption opt;

#define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}

		ADDUSAGE(" ");
		ADDUSAGE("Usage");
		ADDUSAGE(" ");
		ADDUSAGE(" -h  --help             Prints this help ");
		ADDUSAGE(" -i  --infile #         Input snapshot filename ");
		ADDUSAGE("     --logfile #        Log filename [" << logFileName << "]");
		ADDUSAGE("     --dev #            Device ID [" << devID << "]");
		ADDUSAGE("     --renderdev #      Rendering Device ID [" << renderDevID << "]");
		ADDUSAGE(" -t  --dt #             time step [" << timeStep << "]");
		ADDUSAGE(" -T  --tend #           N-body end time [" << tEnd << "]");
		ADDUSAGE(" -I  --iend #           N-body end iteration [" << iterEnd << "]");
		ADDUSAGE(" -e  --eps #            softening (will be squared) [" << eps << "]");
		ADDUSAGE(" -o  --theta #          opening angle (theta) [" <<theta << "]");
		ADDUSAGE("     --snapname #       snapshot base name (N-body time is appended in 000000 format) [" << snapshotFile << "]");
		ADDUSAGE("     --snapiter #       snapshot iteration (N-body time) [" << snapshotIter << "]");
		ADDUSAGE("     --rmdist #         Particle removal distance (-1 to disable) [" << remoDistance << "]");
		ADDUSAGE("     --valueadd #       value to add to the snapshot [" << snapShotAdd << "]");
		ADDUSAGE(" -r  --rebuild #        rebuild tree every # steps [" << rebuild_tree_rate << "]");
		ADDUSAGE("     --reducebodies #   cut down bodies dataset by # factor ");
#ifdef USE_DUST
    ADDUSAGE("     --reducedust #     cut down dust dataset by # factor ");
#endif
#if ENABLE_LOG
    ADDUSAGE("     --log              enable logging ");
    ADDUSAGE("     --prepend-rank     prepend the MPI rank in front of the log-lines ");
#endif
        ADDUSAGE("     --direct           enable N^2 direct gravitation [" << (direct ? "on" : "off") << "]");
#ifdef USE_OPENGL
		ADDUSAGE("     --fullscreen #     set fullscreen mode string");
    ADDUSAGE("     --displayfps       enable on-screen FPS display");
		ADDUSAGE("     --Tglow  #         enable glow @ # Myr [" << TstartGlow << "]");
		ADDUSAGE("     --dTglow  #        reach full brightness in @ # Myr [" << dTstartGlow << "]");
		ADDUSAGE("     --stereo           enable stereo rendering");
#endif


		ADDUSAGE("     --plummer  #      use plummer model with # particles per proc");
#ifdef GALACTICS
		ADDUSAGE("     --milkyway #      use Milky Way model with # particles per proc");
		ADDUSAGE("     --mwfork   #      fork Milky Way generator into # processes [" << nMWfork << "]");
#endif
		ADDUSAGE("     --sphere   #      use spherical model with # particles per proc");
    ADDUSAGE("     --diskmode        use diskmode to read same input file all MPI taks and randomly shuffle its positions");
		ADDUSAGE(" ");


		opt.setFlag( "help" ,   'h');
		opt.setFlag( "diskmode");
		opt.setOption( "infile",  'i');
		opt.setOption( "dt",      't' );
		opt.setOption( "tend",    'T' );
		opt.setOption( "iend",    'I' );
		opt.setOption( "eps",     'e' );
		opt.setOption( "theta",   'o' );
		opt.setOption( "rebuild", 'r' );
    opt.setOption( "plummer");
#ifdef GALACTICS
    opt.setOption( "milkyway");
    opt.setOption( "mwfork");
#endif
    opt.setOption( "sphere");
    opt.setOption( "dev" );
    opt.setOption( "renderdev" );
    opt.setOption( "logfile" );
    opt.setOption( "snapname");
    opt.setOption( "snapiter");
    opt.setOption( "rmdist");
    opt.setOption( "valueadd");
    opt.setOption( "reducebodies");
#ifdef USE_DUST
    opt.setOption( "reducedust");
#endif /* USE_DUST */
#if ENABLE_LOG
    opt.setFlag("log");
    opt.setFlag("prepend-rank");
#endif
    opt.setFlag("direct");
#ifdef USE_OPENGL
    opt.setOption( "fullscreen");
    opt.setOption( "Tglow");
    opt.setOption( "dTglow");
    opt.setFlag("displayfps");
    opt.setFlag("stereo");
#endif

    opt.processCommandArgs( argc, argv );


    if( ! opt.hasOptions()) { /* print usage if no options */
      opt.printUsage();
      exit(0);
    }

    if( opt.getFlag( "help" ) || opt.getFlag( 'h' ) ) 
    {
      opt.printUsage();
      exit(0);
    }

    if (opt.getFlag("direct")) direct = true;
    if (opt.getFlag("displayfps")) displayFPS = true;
    if (opt.getFlag("diskmode")) diskmode = true;
    if(opt.getFlag("stereo"))   stereo = true;

#if ENABLE_LOG
    if (opt.getFlag("log"))           ENABLE_RUNTIME_LOG = true;
    if (opt.getFlag("prepend-rank"))  PREPEND_RANK       = true;
#endif    
    char *optarg = NULL;
    if ((optarg = opt.getValue("infile")))       fileName           = string(optarg);
    if ((optarg = opt.getValue("plummer")))      nPlummer           = atoi(optarg);
    if ((optarg = opt.getValue("milkyway")))     nMilkyWay          = atoi(optarg);
    if ((optarg = opt.getValue("mwfork")))       nMWfork            = atoi(optarg);
    if ((optarg = opt.getValue("sphere")))       nSphere            = atoi(optarg);
    if ((optarg = opt.getValue("logfile")))      logFileName        = string(optarg);
    if ((optarg = opt.getValue("dev")))          devID              = atoi  (optarg);
    renderDevID = devID;
    if ((optarg = opt.getValue("renderdev")))    renderDevID        = atoi  (optarg);
    if ((optarg = opt.getValue("dt")))           timeStep           = (float) atof  (optarg);
    if ((optarg = opt.getValue("tend")))         tEnd               = (float) atof  (optarg);
    if ((optarg = opt.getValue("iend")))         iterEnd            = atoi  (optarg);
    if ((optarg = opt.getValue("eps")))          eps                = (float) atof  (optarg);
    if ((optarg = opt.getValue("theta")))        theta              = (float) atof  (optarg);
    if ((optarg = opt.getValue("snapname")))     snapshotFile       = string(optarg);
    if ((optarg = opt.getValue("snapiter")))     snapshotIter       = (float) atof  (optarg);
    if ((optarg = opt.getValue("rmdist")))       remoDistance       = (float) atof  (optarg);
    if ((optarg = opt.getValue("valueadd")))     snapShotAdd        = atoi  (optarg);
    if ((optarg = opt.getValue("rebuild")))      rebuild_tree_rate  = atoi  (optarg);
    if ((optarg = opt.getValue("reducebodies"))) reduce_bodies_factor = atoi  (optarg);
    if ((optarg = opt.getValue("reducedust")))	 reduce_dust_factor = atoi  (optarg);
#if USE_OPENGL
    if ((optarg = opt.getValue("fullscreen")))	 fullScreenMode     = string(optarg);
    if ((optarg = opt.getValue("Tglow")))	 TstartGlow  = (float)atof(optarg);
    if ((optarg = opt.getValue("dTglow")))	 dTstartGlow  = (float)atof(optarg);
    dTstartGlow = std::max(dTstartGlow, 1.0f);
#endif
    if (fileName.empty() && nPlummer == -1 && nSphere == -1 && nMilkyWay == -1)
    {
      opt.printUsage();
      exit(0);
    }

#undef ADDUSAGE
  }
#endif
  /************** end - command line arguments ********/


  int NTotal, NFirst, NSecond, NThird;
  NTotal = NFirst = NSecond = NThird = 0;

#ifdef USE_OPENGL
  // create OpenGL context first, and register for interop
  initGL(argc, argv, fullScreenMode.c_str(), stereo);
  cudaGLSetGLDevice(devID);
#endif

  initTimers();

  int pid = -1;
#ifdef WIN32
  pid = _getpid();
#else
  pid = (int)getpid();
#endif
  //Used for profiler, note this has to be done before initing to
  //octree otherwise it has no effect...Therefore use pid instead of mpi procId
  char *gpu_prof_log;
  gpu_prof_log=getenv("CUDA_PROFILE_LOG");
  if(gpu_prof_log){
    char tmp[50];
    sprintf(tmp,"process_%d_%s",pid,gpu_prof_log);
#ifdef WIN32
    //        SetEnvironmentVariable("CUDA_PROFILE_LOG", tmp);
#else
    //        setenv("CUDA_PROFILE_LOG",tmp,1);
    LOGF(stderr, "TESTING log on proc: %d val: %s \n", pid, tmp);
#endif
  }


  //Creat the octree class and set the properties
  octree *tree = new octree(argv, devID, theta, eps, snapshotFile, snapshotIter,  timeStep, tEnd, iterEnd, (int)remoDistance, snapShotAdd, rebuild_tree_rate, direct);

  double tStartup = tree->get_time();

  //Get parallel processing information  
  int procId = tree->mpiGetRank();
  int nProcs = tree->mpiGetNProcs();

  if (procId == 0)
  {
    //NOte cant use LOGF here since MPI isnt initialized yet
    cerr << "[INIT]\tUsed settings: \n";
    cerr << "[INIT]\tInput filename " << fileName << endl;
    cerr << "[INIT]\tLog filename " << logFileName << endl;
    cerr << "[INIT]\tTheta: \t\t"             << theta        << "\t\teps: \t\t"          << eps << endl;
    cerr << "[INIT]\tTimestep: \t"          << timeStep     << "\t\ttEnd: \t\t"         << tEnd << endl;
    cerr << "[INIT]\titerEnd: \t" << iterEnd << endl;
    cerr << "[INIT]\tsnapshotFile: \t"      << snapshotFile << "\tsnapshotIter: \t" << snapshotIter << endl;
    cerr << "[INIT]\tInput file: \t"        << fileName     << "\t\tdevID: \t\t"        << devID << endl;
    cerr << "[INIT]\tRemove dist: \t"   << remoDistance << endl;
    cerr << "[INIT]\tSnapshot Addition: \t"  << snapShotAdd << endl;
    cerr << "[INIT]\tRebuild tree every " << rebuild_tree_rate << " timestep\n";


    if( reduce_bodies_factor > 1 )
      cout << "[INIT]\tReduce number of non-dust bodies by " << reduce_bodies_factor << " \n";
    if( reduce_dust_factor > 1 )
      cout << "[INIT]\tReduce number of dust bodies by " << reduce_dust_factor << " \n";

#if ENABLE_LOG
    if (ENABLE_RUNTIME_LOG)
      cerr << "[INIT]\tRuntime logging is ENABLED \n";
    else
      cerr << "[INIT]\tRuntime logging is DISABLED \n";
#endif
    cerr << "[INIT]\tDirect gravitation is " << (direct ? "ENABLED" : "DISABLED") << endl;
#if USE_OPENGL
    cerr << "[INIT]\tTglow = " << TstartGlow << endl;
    cerr << "[INIT]\tdTglow = " << dTstartGlow << endl;
    cerr << "[INIT]\tstereo = " << stereo << endl;
#endif
#ifdef USE_MPI                
    cerr << "[INIT]\tCode is built WITH MPI Support \n";
#else
    cerr << "[INIT]\tCode is built WITHOUT MPI Support \n";
#endif
  }

#ifdef USE_MPI
#if 1
  omp_set_num_threads(16);
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(pthread_self()  , sizeof( cpu_set_t ), &cpuset );


    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);

    int i, set=-1;
    for (i = 0; i < CPU_SETSIZE; i++)
      if (CPU_ISSET(i, &cpuset))
        set = i;
    //    fprintf(stderr,"[Proc: %d ] Thread %d bound to: %d Total cores: %d\n",
    //        procId, tid,  set, num_cores);
  }
#endif




#if 0
  omp_set_num_threads(4);
  //default
  // int cpulist[] = {0,1,2,3,8,9,10,11};
  int cpulist[] = {0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15}; //HA-PACS
  //int cpulist[] = {0,1,2,3,4,5,6,7};
  //int cpulist[] = {0,2,4,6, 8,10,12,14};
  //int cpulist[] = {1,3,5,7, 9,11,13,15};
  //int cpulist[] = {1,9,5,11, 3,7,13,15};
  //int cpulist[] = {1,15,3,13, 2,4,6,8};
  //int cpulist[] = {1,1,1,1, 1,1,1,1};


#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    //int core_id = procId*4+tid;
    int core_id = (procId%4)*4+tid;
    core_id = cpulist[core_id];

    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    //    if (core_id >= num_cores)

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();
    int return_val = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(pthread_self()  , sizeof( cpu_set_t ), &cpuset );

    int i, set=-1;
    for (i = 0; i < CPU_SETSIZE; i++)
      if (CPU_ISSET(i, &cpuset))
        set = i;
    //printf("CPU2: CPU %d\n", i);


    fprintf(stderr,"Binding thread: %d of rank: %d to cpu: %d CHECK: %d Total cores: %d\n",
        tid, procId, core_id, set, num_cores);

  }
#endif
#endif





#if ENABLE_LOG
#ifdef USE_MPI
  PREPEND_RANK_PROCID = procId;
  PREPEND_RANK_NPROCS = nProcs;
#endif
#endif


  if(nProcs > 1)
  {
    logFileName.append("-");

    char buff[16];
    sprintf(buff,"%d-%d", nProcs, procId);
    logFileName.append(buff);
  }

  ofstream logFile(logFileName.c_str());

  tree->set_context(logFile, false); //Do logging to file and enable timing (false = enabled)

  if (nPlummer == -1 && nSphere == -1 && !diskmode && nMilkyWay == -1)
  {
    if(procId == 0)
    {
#ifdef TIPSYOUTPUT
      read_tipsy_file_parallel(bodyPositions, bodyVelocities, bodyIDs, eps, fileName, 
          procId, nProcs, NTotal, NFirst, NSecond, NThird, tree,
          dustPositions, dustVelocities, dustIDs, reduce_bodies_factor, reduce_dust_factor);    

      //      read_generate_cube(bodyPositions, bodyVelocities, bodyIDs, eps, fileName, 
      //                               procId, nProcs, NTotal, NFirst, NSecond, NThird, tree,
      //                              dustPositions, dustVelocities, dustIDs, reduce_bodies_factor, reduce_dust_factor);    

#else
      read_dumbp_file_parallel(bodyPositions, bodyVelocities, bodyIDs, eps, fileName, procId, nProcs, NTotal, NFirst, NSecond, NThird, tree, reduce_bodies_factor);
#endif
    }
    else
    {
      tree->ICRecv(0, bodyPositions, bodyVelocities,  bodyIDs);
    }
  }
  else if(nMilkyWay >= 0)
  {
#ifdef GALACTICS
    if (procId == 0) printf("Using MilkyWay model with n= %d per proc, forked %d times \n", nMilkyWay, nMWfork);
    assert(nMilkyWay > 0);
    assert(nMWfork > 0);
 

#if 0 /* in this setup all particles will be of equal mass (exact number are galactic-depednant)  */
    const float fdisk  = 15.1; 
    const float fbulge = 5.1;   
    const float fhalo  = 242.31; 
#else  /* here, bulge & mw particles have the same mass, but halo particles is 32x heavier */
    const float fdisk  = 15.1; 
    const float fbulge = 5.1; 
    const float fhalo  = 7.5; 
#endif

    const float fsum = fdisk + fhalo + fbulge;

    const int ndisk  = (int)(nMilkyWay * fdisk/fsum);
    const int nbulge = (int)(nMilkyWay * fbulge/fsum);
    const int nhalo  = (int)(nMilkyWay * fhalo/fsum);

    assert(ndisk  > 0);
    assert(nbulge > 0);
    assert(nhalo  > 0);

    const double t0 = tree->get_time();
    const Galactics g(procId, nProcs, ndisk, nbulge, nhalo, nMWfork);
    const double dt = tree->get_time() - t0;
    if (procId == 0)
      printf("  ndisk= %d  nbulge= %d  nhalo= %d :: ntotal= %d in %g sec\n",
          g.get_ndisk(), g.get_nbulge(), g.get_nhalo(), g.get_ntot(), dt);

    const int ntot = g.get_ntot();
    bodyPositions.resize(ntot);
    bodyVelocities.resize(ntot);
    bodyIDs.resize(ntot);
    for (int i= 0; i < ntot; i++)
    {
      assert(!std::isnan(g[i].x));
      assert(!std::isnan(g[i].y));
      assert(!std::isnan(g[i].z));
      assert(g[i].mass > 0.0);
      bodyIDs[i] = g[i].id;

      bodyPositions[i].x = g[i].x;
      bodyPositions[i].y = g[i].y;
      bodyPositions[i].z = g[i].z;
      bodyPositions[i].w = g[i].mass * 1.0/(double)nProcs;
      
      assert(!std::isnan(g[i].vx));
      assert(!std::isnan(g[i].vy));
      assert(!std::isnan(g[i].vz));

      bodyVelocities[i].x = g[i].vx;
      bodyVelocities[i].y = g[i].vy;
      bodyVelocities[i].z = g[i].vz;
      bodyVelocities[i].w = 0.0;
    }
#else
    assert(0);
#endif
  }
  else if(nPlummer >= 0)
  {
    if (procId == 0) printf("Using plummer model with n= %d per proc \n", nPlummer);
    assert(nPlummer > 0);
    const int seed = 19810614 + procId;
    const Plummer m(nPlummer, procId, seed);
    bodyPositions.resize(nPlummer);
    bodyVelocities.resize(nPlummer);
    bodyIDs.resize(nPlummer);
    for (int i= 0; i < nPlummer; i++)
    {

      assert(!std::isnan(m.pos[i].x));
      assert(!std::isnan(m.pos[i].y));
      assert(!std::isnan(m.pos[i].z));
      assert(m.mass[i] > 0.0);
      bodyIDs[i]   = nPlummer*procId + i;

      bodyPositions[i].x = m.pos[i].x;
      bodyPositions[i].y = m.pos[i].y;
      bodyPositions[i].z = m.pos[i].z;
      bodyPositions[i].w = m.mass[i] * 1.0/nProcs;

      bodyVelocities[i].x = m.vel[i].x;
      bodyVelocities[i].y = m.vel[i].y;
      bodyVelocities[i].z = m.vel[i].z;
      bodyVelocities[i].w = 0;
    }
  }
  else if (nSphere >= 0)
  {
    //Sphere
    if (procId == 0) printf("Using Spherical model with n= %d per proc \n", nSphere);
    assert(nSphere >= 0);
    bodyPositions.resize(nSphere);
    bodyVelocities.resize(nSphere);
    bodyIDs.resize(nSphere);

    srand48(procId+19840501);

    /* generate uniform sphere */
    int np = 0;
    while (np < nSphere)
    {
      const double x = 2.0*drand48()-1.0;
      const double y = 2.0*drand48()-1.0;
      const double z = 2.0*drand48()-1.0;
      const double r2 = x*x+y*y+z*z;
      if (r2 < 1)
      {
        bodyIDs[np]   = nSphere*procId + np;

        bodyPositions[np].x = x;
        bodyPositions[np].y = y;
        bodyPositions[np].z = z;
        bodyPositions[np].w = (1.0/nSphere) * 1.0/nProcs;

        bodyVelocities[np].x = 0;
        bodyVelocities[np].y = 0;
        bodyVelocities[np].z = 0;
        bodyVelocities[np].w = 0;
        np++;
      }//if
    }//while
  }//else
  else if (diskmode)
  {
    if (procId == 0) printf("Using diskmode with filename %s\n", fileName.c_str());
    const int seed = procId+19840501;
    srand48(seed);
    const DiskShuffle disk(fileName);
    const int np = disk.get_ntot();
    bodyPositions.resize(np);
    bodyVelocities.resize(np);
    bodyIDs.resize(np);
    for (int i= 0; i < np; i++)
    {
      bodyIDs[i]   = np*procId + i;

      bodyPositions[i].x = disk.pos(i).x;
      bodyPositions[i].y = disk.pos (i).y;
      bodyPositions[i].z = disk.pos (i).z;
      bodyPositions[i].w = disk.mass(i) * 1.0/nProcs;

      bodyVelocities[i].x = disk.vel(i).x;
      bodyVelocities[i].y = disk.vel(i).y;
      bodyVelocities[i].z = disk.vel(i).z;
      bodyVelocities[i].w = 0;
    }
  }
  else
    assert(0);

  tree->mpiSync();


#ifdef TIPSYOUTPUT
  LOGF(stderr, " t_current = %g\n", tree->get_t_current());
#endif


  //#define SETUP_MERGER
#ifdef SETUP_MERGER
  vector<real4> bodyPositions2;
  vector<real4> bodyVelocities2;
  vector<int>   bodyIDs2;  

  bodyPositions2.insert(bodyPositions2.begin(),   bodyPositions.begin(),  bodyPositions.end());
  bodyVelocities2.insert(bodyVelocities2.begin(), bodyVelocities.begin(), bodyVelocities.end());
  bodyIDs2.insert(bodyIDs2.begin(), bodyIDs.begin(), bodyIDs.end());


  setupMergerModel(bodyPositions,  bodyVelocities,  bodyIDs,
      bodyPositions2, bodyVelocities2, bodyIDs2);

  NTotal *= 2;
  NFirst *= 2;
  NSecond *= 2;
  NThird *= 2;
#endif


  //Set the properties of the data set, it only is really used by process 0, which does the 
  //actual file I/O  
  tree->setDataSetProperties(NTotal, NFirst, NSecond, NThird);

  if(procId == 0)  
    LOG("Dataset particle information: Ntotal: %d\tNFirst: %d\tNSecond: %d\tNThird: %d \n",
        NTotal, NFirst, NSecond, NThird);


  //Sanity check for standard plummer spheres
  double mass = 0, totalMass;
  for(unsigned int i=0; i < bodyPositions.size(); i++)
  {
    mass += bodyPositions[i].w;
  }

  tree->load_kernels();

#ifdef USE_MPI
  MPI_Reduce(&mass,&totalMass,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
#else
  totalMass = mass;
#endif

  if(procId == 0)   LOGF(stderr, "Combined Mass: %f \tNTotal: %d \n", totalMass, NTotal);

  LOG("Starting! Bootup time: %lg \n", tree->get_time()-tStartup);


  double t0 = tree->get_time();

  tree->localTree.setN((int)bodyPositions.size());
  tree->allocateParticleMemory(tree->localTree);

  //Load data onto the device
  for(uint i=0; i < bodyPositions.size(); i++)
  {
    tree->localTree.bodies_pos[i] = bodyPositions[i];
    tree->localTree.bodies_vel[i] = bodyVelocities[i];
    tree->localTree.bodies_ids[i] = bodyIDs[i];

    tree->localTree.bodies_Ppos[i] = bodyPositions[i];
    tree->localTree.bodies_Pvel[i] = bodyVelocities[i];
    tree->localTree.bodies_time[i] = make_float2(tree->get_t_current(), tree->get_t_current());
  }

  tree->localTree.bodies_time.h2d();
  tree->localTree.bodies_pos.h2d();
  tree->localTree.bodies_vel.h2d();
  tree->localTree.bodies_Ppos.h2d();
  tree->localTree.bodies_Pvel.h2d();
  tree->localTree.bodies_ids.h2d();

  //fprintf(stderr,"Send data to device proc: %d \n", procId);
  //  tree->devContext.writeLogEvent("Send data to device\n");



#ifdef USE_MPI
  //Use sampling particles, determine frequency
  tree->mpiSumParticleCount(tree->localTree.n); //Determine initial frequency
#endif


  //If required set the dust particles
#ifdef USE_DUST
  if( (int)dustPositions.size() > 0)
  {
    LOGF(stderr, "Allocating dust properties for %d dust particles \n",
        (int)dustPositions.size());   
    tree->localTree.setNDust((int)dustPositions.size());
    tree->allocateDustMemory(tree->localTree);

    //Load dust data onto the device
    for(uint i=0; i < dustPositions.size(); i++)
    {
      tree->localTree.dust_pos[i] = dustPositions[i];
      tree->localTree.dust_vel[i] = dustVelocities[i];
      tree->localTree.dust_ids[i] = dustIDs[i];
    }

    tree->localTree.dust_pos.h2d();
    tree->localTree.dust_vel.h2d();
    tree->localTree.dust_ids.h2d();    
  }
#endif //ifdef USE_DUST


#ifdef USE_MPI
  //Startup the OMP threads
  omp_set_num_threads(4);
#endif


  //Start the integration
#ifdef USE_OPENGL
  octree::IterationData idata;
  initAppRenderer(argc, argv, tree, idata, displayFPS, stereo);
  LOG("Finished!!! Took in total: %lg sec\n", tree->get_time()-t0);
#else
  tree->mpiSync();
  if (procId==0)
    fprintf(stderr, " Starting iterating\n");
  tree->mpiSync();
  tree->iterate(); 

  LOG("Finished!!! Took in total: %lg sec\n", tree->get_time()-t0);

  logFile.close();

#ifdef USE_MPI
  MPI_Finalize();
#endif

  if(tree->procId == 0)
  {
    LOGF(stderr, "TOTAL:   Time spent between the start of 'iterate' and the final time-step (very first step is not accounted)\n",0);
    LOGF(stderr, "Grav:    Time spent to compute gravity, including communication (wall-clock time)\n",0);
    LOGF(stderr, "GPUgrav: Time spent ON the GPU to compute local and LET gravity\n",0);
    LOGF(stderr, "LET Com: Time spent in exchanging and building LET data\n",0);
    LOGF(stderr, "Build:   Time spent in constructing the tree (incl sorting, making groups, etc.)\n",0);
    LOGF(stderr, "Domain:  Time spent in computing new domain decomposition and exchanging particles between nodes.\n",0);
    LOGF(stderr, "Wait:    Time spent in waiting on other processes after the gravity part.\n",0);
  }


  delete tree;
  tree = NULL;
#endif

  displayTimers();
  return 0;
}
