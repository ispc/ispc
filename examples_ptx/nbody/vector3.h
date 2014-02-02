#pragma once

#include <iostream>
#include <fstream>
#include <cmath>

template <class REAL> struct vector3{
  public:
    REAL x, y, z;

    vector3(){
      x = y = z = REAL(0);
    }
    vector3(const REAL &r){
      x = y = z = r;
    }
    vector3(const REAL &_x, const REAL &_y, const REAL &_z){
      x = _x;  y = _y;  z = _z;
    }
    vector3(const REAL *p){
      x = p[0]; y = p[1]; z = p[2];
    }
    ~vector3(){}

    REAL &operator [](int i){
      return (&x)[i];
    }
    const REAL &operator [](int i) const{
      return (&x)[i];
    }
    template <class real> 
      operator vector3<real> () const {
        return vector3<real> (real(x), real(y), real(z));
      }
    operator REAL *(){
      return &x;
    }
    REAL (*toPointer())[3]{
      return (REAL (*)[3])&x;
    }
    typedef REAL (*pArrayOfReal3)[3];
    operator pArrayOfReal3(){
      return toPointer();
    }

    void outv(std::ostream &ofs = std::cout) const{
      ofs << "(" << x << ", " << y << ", " << z << ")" << std::endl;
    }
    bool are_numbers () const{
      // returns false if *this has (a) NaN member(s)
      return (norm2() >= REAL(0));
    }

    REAL norm2() const{
      return (*this)*(*this);
    }
    REAL abs() const{
      return std::sqrt(norm2());
    }

    friend std::ostream &operator << (std::ostream &ofs, const vector3<REAL> &v){
      ofs << v.x << " " << v.y << " " << v.z;
      return ofs;
    }
    friend std::istream &operator >> (std::istream &ifs, vector3<REAL> &v){
      ifs >> v.x >> v.y >> v.z;
      return ifs;
    }
    const vector3<REAL> operator + (const vector3<REAL> &v) const{
      return vector3<REAL> (x+v.x, y+v.y, z+v.z);
    }
    const inline vector3<REAL> operator - (const vector3<REAL> &v) const{
      return vector3<REAL> (x-v.x, y-v.y, z-v.z);
    }
    const vector3<REAL> operator * (const REAL &s) const{
      return vector3<REAL> (x*s, y*s, z*s);
    }
    friend const vector3<REAL> operator * (const REAL &s, const vector3<REAL> &v){
      return v*s;
    }
    // dot product
    const inline REAL operator * (const vector3<REAL> &v) const{
      return (x*v.x + y*v.y + z*v.z);
    }
    // vector product
    const vector3<REAL> operator % (const vector3<REAL> &v) const{
      return vector3<REAL> (
          y*v.z - z*v.y, 
          z*v.x - x*v.z, 
          x*v.y - y*v.x);
    }
    const vector3<REAL> operator / (const REAL &s) const{
      REAL r = REAL(1)/s;
      return (*this)*r;
    }
    const vector3<REAL> operator = (const vector3<REAL> &v){
      x = v.x; y=v.y; z=v.z;
      return *this;
    }

    const vector3<REAL> operator - (){
      return vector3<REAL> (-x, -y, -z);
    }
    const vector3<REAL> &operator += (const vector3<REAL> &v){
      *this = *this + v;
      return *this;
    }
    const vector3<REAL> &operator -= (const vector3<REAL> &v){
      *this = *this - v;
      return *this;
    }
    const vector3<REAL> &operator *= (const REAL &s){
      *this = *this * s;
      return *this;
    }
    const vector3<REAL> &operator /= (const REAL &s){
      *this = *this / s;
      return *this;
    }

    friend const vector3<REAL> maxeach (const vector3<REAL> &a, const vector3<REAL> &b){
      return vector3<REAL> (std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
    }
    friend const vector3<REAL> mineach (const vector3<REAL> &a, const vector3<REAL> &b){
      return vector3<REAL> (std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
    }
    const vector3<REAL> abseach(){
      return vector3<REAL> (std::fabs(x), std::fabs(y), std::fabs(z));
    }
};

typedef vector3<double> dvec3;
typedef vector3<float>  fvec3;


