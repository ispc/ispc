#ifdef __NVPTX__  
#define blockIndex0 blockIndex0()
#define blockIndex1 blockIndex1()
#define vectorWidth warpSize()
#define vectorIndex laneIndex()
#else
#define blockIndex0 taskIndex0
#define blockIndex1 taskIndex1
#define vectorWidth programCount
#define vectorIndex programIndex
#endif

#if 0

varying float mem_private[100];
uniform float mem_shared [100];

#else

  static inline int
mandel(float c_re, float c_im, int count) 
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i) {
    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re*z_re - z_im*z_im;
    float new_im = 2.f * z_re * z_im;
    unmasked {
      z_re = c_re + new_re;
      z_im = c_im + new_im;
    }
  }

  return i;
}


task void
mandelbrot_scanline(
    uniform float x0, uniform float dx, 
    uniform float y0, uniform float dy,
    uniform int width, uniform int height, 
    uniform int xspan, uniform int yspan,
    uniform int maxIterations, uniform int output[]) 
{
  const uniform int xstart = blockIndex0 * xspan;
  const uniform int xend   = min(xstart  + xspan, width);

  const uniform int ystart = blockIndex1 * yspan;
  const uniform int yend   = min(ystart  + yspan, height);

//  assert(xspan >= vectorWidth);

  for (uniform int yi = ystart; yi < yend; yi++)
    for (uniform int xi = xstart; xi < xend; xi += vectorWidth)
    {
      const float x = x0 + (xi + vectorIndex) * dx;
      const float y = y0 +  yi              * dy;

      const int res = mandel(x,y,maxIterations);
      const int index = yi * width + (xi + vectorIndex);
      if (xi + vectorIndex < xend)
        output[index] = res;
    }
}

#if 1
export void
mandelbrot_ispc(uniform float x0, uniform float y0, 
                uniform float x1, uniform float y1,
                uniform int width, uniform int height, 
                uniform int maxIterations, uniform int output[]) {
    uniform float dx = (x1 - x0) / width;
    uniform float dy = (y1 - y0) / height;
    const uniform int xspan = 32;  /* make sure it is big enough to avoid false-sharing */
    const uniform int yspan = 4; 

    launch [width/xspan, height/yspan]
      mandelbrot_scanline(x0, dx, y0, dy, width, height, xspan, yspan,
                          maxIterations, output);
#if 0
    launch [width/xspan] [height/yspan]
      mandelbrot_scanline(x0, dx, y0, dy, width, height, xspan, yspan,
                          maxIterations, output);
#endif
}
#endif
#endif
