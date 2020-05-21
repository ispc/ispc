// RUN: %{ispc} --target=host %s > %t 2>&1
inline float foo(const float<3> &v) { return v.x + v.y + v.z; }

inline float<3> bar(const float<3> &v) { return 2.0f * v * foo(v); }
