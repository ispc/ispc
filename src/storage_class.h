#pragma once

namespace ispc {

enum class StorageClass {
    None,
    Extern,
    Static,
    Typedef,
    ExternC
};

constexpr const char *ToString(StorageClass sc) noexcept {
    switch (sc) {
    case StorageClass::None:
        break;
    case StorageClass::Extern:
        return "extern";
    case StorageClass::Static:
        return "static";
    case StorageClass::Typedef:
        return "typedef";
    case StorageClass::ExternC:
        return "extern \"C\"";
    }
    return "";
}

} // namespace ispc
