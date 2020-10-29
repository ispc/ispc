#pragma once

#include <cstddef>

namespace ispc {

struct SourcePos final {
    std::size_t first_line;
    std::size_t first_column;
    std::size_t last_line;
    std::size_t last_column;
};

} // namespace ispc
