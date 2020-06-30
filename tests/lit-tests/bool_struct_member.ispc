// RUN: %{ispc} %s --target=host > %t 2>&1
// RUN: FileCheck --input-file=%t %s
// CHECK-NOT: Please file a bug report at https://github.com/ispc/ispc/issues
struct BoolHelper {
    bool *bHelper;
};

void CopyBool(const uniform BoolHelper Helper) { const uniform bool bHelper = Helper.bHelper[5]; }
