static inline int mandel(float c_re, float c_im, uniform int count)
{
  float z_re = c_re, z_im = c_im;
  int j = 0;
  for (uniform int i = 0; i < count; ++i) {
    bool done = (z_re * z_re + z_im * z_im > 4.f);
    if (all(done))
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;

    z_re = c_re + new_re;
    z_im = c_im + new_im;

    if (!done)
      j++;
  }

  return j;
}

task void mandelbrot_tile(uniform float x0,
                          uniform float dx,
                          uniform float y0,
                          uniform float dy,
                          uniform int width,
                          uniform int height,
                          uniform int tile_size,
                          uniform int maxIterations,
                          uniform int output[])
{
  uniform const int x_begin = taskIndex0 * tile_size;
  uniform const int y_begin = taskIndex1 * tile_size;

  uniform const int x_edge = x_begin + tile_size;
  uniform const int y_edge = y_begin + tile_size;

  uniform const int x_end = min(x_edge, width);
  uniform const int y_end = min(y_edge, height);

  foreach (yi = y_begin... y_end, xi = x_begin... x_end) {
    const float x = x0 + xi * dx;
    const float y = y0 + yi * dy;

    const int index = yi * width + xi;
    output[index]   = mandel(x, y, maxIterations);
  }
}