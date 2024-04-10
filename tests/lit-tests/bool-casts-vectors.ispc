// RUN: %{ispc} --target=host --nowrap --nostdlib -O2 --emit-llvm-text %s -o - | FileCheck %s

// CHECK-LABEL: @to_float16___unh_3C_4_3E_unh_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = fcmp olt <4 x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = uitofp <4 x i1> [[CMP]] to <4 x [[TYPE]]>
// CHECK-NEXT: ret <4 x [[TYPE]]> [[CAST]]

uniform float16<4> to_float16(uniform float16<4> a, uniform float16<4> b) { return a < b; }

// CHECK-LABEL: @to_float___unf_3C_4_3E_unf_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = fcmp olt <4 x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = uitofp <4 x i1> [[CMP]] to <4 x [[TYPE]]>
// CHECK-NEXT: ret <4 x [[TYPE]]> [[CAST]]

uniform float<4> to_float(uniform float<4> a, uniform float<4> b) { return a < b; }

// CHECK-LABEL: @to_double___und_3C_4_3E_und_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = fcmp olt <4 x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = uitofp <4 x i1> [[CMP]] to <4 x [[TYPE]]>
// CHECK-NEXT: ret <4 x [[TYPE]]> [[CAST]]

uniform double<4> to_double(uniform double<4> a, uniform double<4> b) { return a < b; }

// CHECK-LABEL: @to_int8___unt_3C_4_3E_unt_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp slt <4 x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <4 x i1> [[CMP]] to <4 x [[TYPE]]>
// CHECK-NEXT: ret <4 x [[TYPE]]> [[CAST]]

uniform int8<4> to_int8(uniform int8<4> a, uniform int8<4> b) { return a < b; }

// CHECK-LABEL: @to_uint8___unT_3C_4_3E_unT_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp ugt <4 x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <4 x i1> [[CMP]] to <4 x [[TYPE]]>
// CHECK-NEXT: ret <4 x [[TYPE]]> [[CAST]]

uniform uint8<4> to_uint8(uniform uint8<4> a, uniform uint8<4> b) { return a > b; }

// CHECK-LABEL: @to_int16___uns_3C_4_3E_uns_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp slt <4 x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <4 x i1> [[CMP]] to <4 x [[TYPE]]>
// CHECK-NEXT: ret <4 x [[TYPE]]> [[CAST]]

uniform int16<4> to_int16(uniform int16<4> a, uniform int16<4> b) { return a < b; }

// CHECK-LABEL: @to_uint16___unS_3C_4_3E_unS_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp ugt <4 x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <4 x i1> [[CMP]] to <4 x [[TYPE]]>
// CHECK-NEXT: ret <4 x [[TYPE]]> [[CAST]]

uniform uint16<4> to_uint16(uniform uint16<4> a, uniform uint16<4> b) { return a > b; }

// CHECK-LABEL: @to_int32___uni_3C_4_3E_uni_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp slt <4 x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <4 x i1> [[CMP]] to <4 x [[TYPE]]>
// CHECK-NEXT: ret <4 x [[TYPE]]> [[CAST]]

uniform int32<4> to_int32(uniform int32<4> a, uniform int32<4> b) { return a < b; }

// CHECK-LABEL: @to_uint32___unu_3C_4_3E_unu_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp ugt <4 x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <4 x i1> [[CMP]] to <4 x [[TYPE]]>
// CHECK-NEXT: ret <4 x [[TYPE]]> [[CAST]]

uniform uint32<4> to_uint32(uniform uint32<4> a, uniform uint32<4> b) { return a > b; }

// CHECK-LABEL: @to_int64___unI_3C_4_3E_unI_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp slt <4 x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <4 x i1> [[CMP]] to <4 x [[TYPE]]>
// CHECK-NEXT: ret <4 x [[TYPE]]> [[CAST]]

uniform int64<4> to_int64(uniform int64<4> a, uniform int64<4> b) { return a < b; }

// CHECK-LABEL: @to_uint64___unU_3C_4_3E_unU_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[CMP:%.*]] = icmp ugt <4 x [[TYPE:.*]]> %a, %b
// CHECK-NEXT: [[CAST:%.*]] = zext <4 x i1> [[CMP]] to <4 x [[TYPE]]>
// CHECK-NEXT: ret <4 x [[TYPE]]> [[CAST]]

uniform uint64<4> to_uint64(uniform uint64<4> a, uniform uint64<4> b) { return a > b; }

// CHECK-LABEL: @to_bool___unb_3C_4_3E_unb_3C_4_3E_(
// CHECK-NEXT: allocas:
// CHECK-NEXT: [[XOR:%.*]] = xor <4 x i1> %a, %b
// CHECK-NEXT: [[NE:%.*]] = xor <4 x i1> [[XOR]], <i1 true, i1 true, i1 true, i1 true>
// CHECK-NEXT: ret <4 x i1> [[NE]]

uniform bool<4> to_bool(uniform bool<4> a, uniform bool<4> b) { return a == b; }
