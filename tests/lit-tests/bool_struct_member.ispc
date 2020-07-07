// RUN: %{ispc} %s --target=host --nostdlib
struct BoolHelper {
    bool *bHelper;
};

void CopyBool(const uniform BoolHelper Helper) { const uniform bool bHelper = Helper.bHelper[5]; }
