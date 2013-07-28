export void saxpy(const uniform float a, const uniform float x_[], const uniform float y_[], uniform float z_[], const uniform int n)
{
  foreach (i = 0 ... n)
  {
     const float x = x_[i];
     const float y = y_[i];
     const float z = y + a*x;
     z_[i] = rsqrt(z);
  }
}
