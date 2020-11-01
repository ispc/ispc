#pragma once

#include <gtest/gtest.h>

namespace ispc {

::testing::Test *MakeDiagnosticTest(const char *sourcePath,
                                    const char *expectedErrorPath);

} // namespace ispc
