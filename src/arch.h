#pragma once

#include <optional>
#include <string_view>

namespace ispc {

enum class Arch { none, x86, x86_64, arm, aarch64, wasm32, genx32, genx64, error };

constexpr std::optional<Arch> ParseArch(const std::string_view &arch) noexcept {
    if (arch == "x86") {
        return Arch::x86;
    } else if (arch == "x86_64" || arch == "x86-64") {
        return Arch::x86_64;
    } else if (arch == "arm") {
        return Arch::arm;
    } else if (arch == "aarch64") {
        return Arch::aarch64;
    } else if (arch == "wasm32") {
        return Arch::wasm32;
    } else if (arch == "genx32") {
        return Arch::genx32;
    } else if (arch == "genx64") {
        return Arch::genx64;
    }
    return std::nullopt;
}

constexpr const char *ArchToString(Arch arch) noexcept {
    switch (arch) {
    case Arch::none:
    case Arch::error:
        break;
    case Arch::x86:
        return "x86";
    case Arch::x86_64:
        return "x86-64";
    case Arch::arm:
        return "arm";
    case Arch::aarch64:
        return "aarch64";
    case Arch::wasm32:
        return "wasm32";
    case Arch::genx32:
        return "genx32";
    case Arch::genx64:
        return "genx64";
    }
    return "";
}

} // namespace ispc
