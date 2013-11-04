extern task void
mandelbrot_scanline(
    uniform float x0, uniform float dx, 
    uniform float y0, uniform float dy,
    uniform int width, uniform int height, 
    uniform int xspan, uniform int yspan,
    uniform int maxIterations, uniform int output[]);

export void
mandelbrot_ispc(uniform float x0, uniform float y0, 
                uniform float x1, uniform float y1,
                uniform int width, uniform int height, 
                uniform int maxIterations, uniform int output[]) {
    uniform float dx = (x1 - x0) / width;
    uniform float dy = (y1 - y0) / height;
    const uniform int xspan = 16;  /* make sure it is big enough to avoid false-sharing */
    const uniform int yspan = 16; 

    launch [width/xspan, height/yspan]
      mandelbrot_scanline(x0, dx, y0, dy, width, height, xspan, yspan,
                          maxIterations, output);
}
