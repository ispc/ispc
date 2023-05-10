// The test ensures that the nsw bit is suppressed for Xe targets in LLVMIR, unless specified on.
// RUN: %{ispc} %s --emit-llvm-text --target=xehpg-x8 -o - | FileCheck %s --check-prefix=CHECK_NONSW
// RUN: %{ispc} %s --emit-llvm-text --target=xehpc-x16 -o - | FileCheck %s --check-prefix=CHECK_NONSW
// RUN: %{ispc} %s --emit-llvm-text --target=xelp-x8 -o - | FileCheck %s --check-prefix=CHECK_NONSW
// RUN: %{ispc} %s --emit-llvm-text --target=xehpg-x8 -o - --no-wrap-signed-int | FileCheck %s --check-prefix=CHECK_NSW
// RUN: %{ispc} %s --emit-llvm-text --target=xehpc-x16 -o - --no-wrap-signed-int | FileCheck %s --check-prefix=CHECK_NSW
// RUN: %{ispc} %s --emit-llvm-text --target=xelp-x8 -o - --no-wrap-signed-int | FileCheck %s --check-prefix=CHECK_NSW
// RUN: %{ispc} %s --emit-llvm-text --target=xehpg-x8 -o - --wrap-signed-int | FileCheck %s --check-prefix=CHECK_NONSW
// RUN: %{ispc} %s --emit-llvm-text --target=xehpc-x16 -o - --wrap-signed-int | FileCheck %s --check-prefix=CHECK_NONSW
// RUN: %{ispc} %s --emit-llvm-text --target=xelp-x8 -o - --wrap-signed-int | FileCheck %s --check-prefix=CHECK_NONSW

// REQUIRES: XE_ENABLED


// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i64> @mul_64
// CHECK_NONSW-NOT:           %{{.*}}= mul{{.*}}nsw{{.*}}i64
// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i32> @mul_32
// CHECK_NONSW-NOT:           %{{.*}}= mul{{.*}}nsw{{.*}}i32
// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i16> @mul_16
// CHECK_NONSW-NOT:           %{{.*}}= mul{{.*}}nsw{{.*}}i16
// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i8> @mul_8
// CHECK_NONSW-NOT:           %{{.*}}= mul{{.*}}nsw{{.*}}i8

// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i64> @mul_64
// CHECK_NSW-COUNT-1:         %{{.*}}= mul{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i32> @mul_32
// CHECK_NSW-COUNT-1:         %{{.*}}= mul{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i16> @mul_16
// CHECK_NSW-COUNT-1:         %{{.*}}= mul{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i8> @mul_8
// CHECK_NSW-COUNT-1:         %{{.*}}= mul{{.*}}nsw{{.*}}i8

unmasked int64 mul_64(int64 a, int64 b) { return a * b; }

unmasked int32 mul_32(int32 a, int32 b) { return a * b; }

unmasked int16 mul_16(int16 a, int16 b) { return a * b; }

unmasked int8 mul_8(int8 a, int8 b) { return a * b; }


// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i64> @add_64
// CHECK_NONSW-NOT:           %{{.*}}= add{{.*}}nsw{{.*}}i64
// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i32> @add_32
// CHECK_NONSW-NOT:           %{{.*}}= add{{.*}}nsw{{.*}}i32
// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i16> @add_16
// CHECK_NONSW-NOT:           %{{.*}}= add{{.*}}nsw{{.*}}i16
// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i8> @add_8
// CHECK_NONSW-NOT:           %{{.*}}= add{{.*}}nsw{{.*}}i8

// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i64> @add_64
// CHECK_NSW-COUNT-1:         %{{.*}}= add{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i32> @add_32
// CHECK_NSW-COUNT-1:         %{{.*}}= add{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i16> @add_16
// CHECK_NSW-COUNT-1:         %{{.*}}= add{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i8> @add_8
// CHECK_NSW-COUNT-1:         %{{.*}}= add{{.*}}nsw{{.*}}i8

unmasked int64 add_64(int64 a, int64 b) { return a + b; }

unmasked int32 add_32(int32 a, int32 b) { return a + b; }

unmasked int16 add_16(int16 a, int16 b) { return a + b; }

unmasked int8 add_8(int8 a, int8 b) { return a + b; }


// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i64> @sub_64
// CHECK_NONSW-NOT:           %{{.*}}= sub{{.*}}nsw{{.*}}i64
// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i32> @sub_32
// CHECK_NONSW-NOT:           %{{.*}}= sub{{.*}}nsw{{.*}}i32
// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i16> @sub_16
// CHECK_NONSW-NOT:           %{{.*}}= sub{{.*}}nsw{{.*}}i16
// CHECK_NONSW-LABEL:       define{{( spir_func)?}} <{{[0-9]*}} x i8> @sub_8
// CHECK_NONSW-NOT:           %{{.*}}= sub{{.*}}nsw{{.*}}i8

// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i64> @sub_64
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i32> @sub_32
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i16> @sub_16
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define{{( spir_func)?}} <{{[0-9]*}} x i8> @sub_8
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i8

unmasked int64 sub_64(int64 a, int64 b) { return a - b; }

unmasked int32 sub_32(int32 a, int32 b) { return a - b; }

unmasked int16 sub_16(int16 a, int16 b) { return a - b; }

unmasked int8 sub_8(int8 a, int8 b) { return a - b; }
