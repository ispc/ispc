extern "C" static inline int __device__ mandel___vyfvyfvyi_(float c_re, float c_im, int count) {}
extern "C" void __global__ mandelbrot_scanline___unfunfunfunfuniuniuniuniuniun_3C_uni_3E_( float x0,  float dx, 
    float y0,  float dy,
    int width,  int height, 
    int xspan,  int yspan,
    int maxIterations,  int output[]) {}
extern "C" void __global__ mandelbrot_ispc___unfunfunfunfuniuniuniun_3C_uni_3E_( float x0,  float y0, 
    float x1,  float y1,
    int width,  int height, 
    int maxIterations,  int output[]) { }

extern "C"
void mandelbrot_ispc(float x0, float y0, 
    float x1, float y1,
    int width, int height, 
    int maxIterations, int output[])
{
  mandelbrot_ispc___unfunfunfunfuniuniuniun_3C_uni_3E_<<<1,32>>>
    (x0,y0,x1,y1,width,height,maxIterations,output);
  cudaDeviceSynchronize();
}

