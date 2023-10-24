// RUN: %{ispc} --target=host --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s

// CHECK-LABEL: @to_float16___vyhvyh(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = fcmp olt <[[WIDTH:.*]] x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = uitofp <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x [[TYPE]]>
// CHECK-NEXT: ret <[[WIDTH]] x [[TYPE]]> [[CAST]]

float16 to_float16(float16 a, float16 b) { return a < b; }

// CHECK-LABEL: @to_float___vyfvyf(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = fcmp olt <[[WIDTH:.*]] x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = uitofp <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x [[TYPE]]>
// CHECK-NEXT: ret <[[WIDTH]] x [[TYPE]]> [[CAST]]

float to_float(float a, float b) { return a < b; }

// CHECK-LABEL: @to_double___vydvyd(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = fcmp olt <[[WIDTH:.*]] x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = uitofp <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x [[TYPE]]>
// CHECK-NEXT: ret <[[WIDTH]] x [[TYPE]]> [[CAST]]

double to_double(double a, double b) { return a < b; }

// CHECK-LABEL: @to_int8___vytvyt(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp slt <[[WIDTH:.*]] x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x [[TYPE]]>
// CHECK-NEXT: ret <[[WIDTH]] x [[TYPE]]> [[CAST]]

int8 to_int8(int8 a, int8 b) { return a < b; }

// CHECK-LABEL: @to_uint8___vyTvyT(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp ugt <[[WIDTH:.*]] x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x [[TYPE]]>
// CHECK-NEXT: ret <[[WIDTH]] x [[TYPE]]> [[CAST]]

uint8 to_uint8(uint8 a, uint8 b) { return a > b; }

// CHECK-LABEL: @to_int16___vysvys(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp slt <[[WIDTH:.*]] x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x [[TYPE]]>
// CHECK-NEXT: ret <[[WIDTH]] x [[TYPE]]> [[CAST]]

int16 to_int16(int16 a, int16 b) { return a < b; }

// CHECK-LABEL: @to_uint16___vySvyS(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp ugt <[[WIDTH:.*]] x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x [[TYPE]]>
// CHECK-NEXT: ret <[[WIDTH]] x [[TYPE]]> [[CAST]]

uint16 to_uint16(uint16 a, uint16 b) { return a > b; }

// CHECK-LABEL: @to_int32___vyivyi(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp slt <[[WIDTH:.*]] x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x [[TYPE]]>
// CHECK-NEXT: ret <[[WIDTH]] x [[TYPE]]> [[CAST]]

int32 to_int32(int32 a, int32 b) { return a < b; }

// CHECK-LABEL: @to_uint32___vyuvyu(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp ugt <[[WIDTH:.*]] x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x [[TYPE]]>
// CHECK-NEXT: ret <[[WIDTH]] x [[TYPE]]> [[CAST]]

uint32 to_uint32(uint32 a, uint32 b) { return a > b; }

// CHECK-LABEL: @to_int64___vyIvyI(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp slt <[[WIDTH:.*]] x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x [[TYPE]]>
// CHECK-NEXT: ret <[[WIDTH]] x [[TYPE]]> [[CAST]]

int64 to_int64(int64 a, int64 b) { return a < b; }

// CHECK-LABEL: @to_uint64___vyUvyU(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp ugt <[[WIDTH:.*]] x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <[[WIDTH]] x i1> [[CMP]] to <[[WIDTH]] x [[TYPE]]>
// CHECK-NEXT: ret <[[WIDTH]] x [[TYPE]]> [[CAST]]

uint64 to_uint64(uint64 a, uint64 b) { return a > b; }

bool to_bool(bool a, bool b) { return a == b; }
