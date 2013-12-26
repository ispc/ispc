typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

extern "C" __device__ void PTXmandelbrot_scanline___UM_unfunfunfunfuniuniuniuniuniun_3C_uni_3E_(
    float,float,float,float,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t);

extern "C"
__global__ void mandelbrot_scanline___UM_unfunfunfunfuniuniuniuniuniun_3C_uni_3E_(
    float param0, 
    float param1, 
    float param2, 
    float param3, 
    uint32_t param4, 
    uint32_t param5, 
    uint32_t param6, 
    uint32_t param7, 
    uint32_t param8, 
    uint64_t param9) 
{
  PTXmandelbrot_scanline___UM_unfunfunfunfuniuniuniuniuniun_3C_uni_3E_(
      param0, param1, param2, param3, param4, param5, param6, param7, param8, param9);
}

extern "C" __device__ void PTXmandelbrot_ispc___unfunfunfunfuniuniuniun_3C_uni_3E_(
	float param0,
	float param1,
	float param2,
	float param3,
	uint32_t param4,
	uint32_t param5,
	uint32_t param6,
	uint64_t param7,
	char param8);

extern "C"
__global__ void mandelbrot_ispc___unfunfunfunfuniuniuniun_3C_uni_3E_(
	float param0,
	float param1,
	float param2,
	float param3,
	uint32_t param4,
	uint32_t param5,
	uint32_t param6,
	uint64_t param7,
	char param8)
{
 PTXmandelbrot_ispc___unfunfunfunfuniuniuniun_3C_uni_3E_(
     param0,param1,param2,param3,param4,param5,param6,param7,param8);
}

extern "C" __device__ void PTXmandelbrot_ispc(
	float param0,
	float param1,
	float param2,
	float param3,
	uint32_t param4,
	uint32_t param5,
	uint32_t param6,
	uint64_t param7);
extern "C"
__global__ void mandelbrot_ispc(
	float param0,
	float param1,
	float param2,
	float param3,
	uint32_t param4,
	uint32_t param5,
	uint32_t param6,
	uint64_t param7)
{
 PTXmandelbrot_ispc(
     param0,param1,param2,param3,param4,param5,param6,param7);
}
