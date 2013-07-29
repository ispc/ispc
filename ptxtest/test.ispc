export void saxpy(const uniform float a, const uniform float x_[], const uniform float y_[], uniform float z_[], const uniform int n)
{
  foreach (i = 0 ... n)
  {
     const double x = x_[i];
     const double y = y_[i];
     const double dz = y + a*x;
     const double dz1 = 1.0/sqrt(dz);
     const float z = dz1;
     z_[i] = rsqrt(z);
  }
}
