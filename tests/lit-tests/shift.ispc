// RUN: %{ispc} %s --target=host --emit-llvm-text -o - | FileCheck %s

typedef uniform unsigned int64 uint64_t;
typedef uniform          int8  int8_t;


uint64_t Func(const uint64_t val) {
	int8_t bits = 0;
	while ((val >> bits) >= 128) { bits += 7; } /* Count number of set bits in 7-bit slices */
	const uint64_t flag = (!bits) << 7; /* This should be set to 128 */
	return flag;
}

// CHECK: ret i64 128
/* This should return 128 */
uint64_t Test() {
    return Func(24);
}
