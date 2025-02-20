// Test here short vec template instantiation with short vector of different types.

// RUN: %{ispc} %s --nowrap --nostdlib --target=host --ast-dump -o %t.o 2>&1 | FileCheck %s

template <typename T, uint N> T<N> same(T<N> a) { return a; }

// CHECK: (instantiation <varying bool, 2>) Function {{.*}}  [ varying bool<2>(varying bool<2> a)] "same"
bool<2> test_bool(bool<2> a) { return same(a); }

// CHECK: (instantiation <varying float, 7>) Function {{.*}}  [ varying float<7>(varying float<7> a)] "same
float<7> test_float(float<7> a) { return same(a); }

// CHECK: (instantiation <varying double, 3>) Function {{.*}}  [ varying double<3>(varying double<3> a)] "same"
double<3> test_double(double<3> a) { return same(a); }

// CHECK: (instantiation <varying float16, 4>) Function {{.*}}  [ varying float16<4>(varying float16<4> a)] "same"
float16<4> test_float16(float16<4> a) { return same(a); }

// CHECK: (instantiation <varying int64, 5>) Function {{.*}}  [ varying int64<5>(varying int64<5> a)] "same"
int64<5> test_int64(int64<5> a) { return same(a); }

// CHECK: (instantiation <varying int32, 6>) Function {{.*}}  [ varying int32<6>(varying int32<6> a)] "same"
int32<6> test_int32(int32<6> a) { return same(a); }

// CHECK: (instantiation <varying int16, 8>) Function {{.*}}  [ varying int16<8>(varying int16<8> a)] "same"
int16<8> test_int16(int16<8> a) { return same(a); }

// CHECK: (instantiation <varying int8, 9>) Function {{.*}}  [ varying int8<9>(varying int8<9> a)] "same"
int8<9> test_int8(int8<9> a) { return same(a); }
