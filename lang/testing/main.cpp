#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>

#include "diagnostic_test.h"

#ifndef TESTING_DIR
#define TESTING_DIR "."
#endif

constexpr const char *GetDiagnosticTestPath() noexcept {
    return TESTING_DIR "/diagnostics/";
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);

  auto diagnosticTestDir =
      std::filesystem::directory_iterator(GetDiagnosticTestPath());

  for (auto test : diagnosticTestDir) {

    std::filesystem::path path(test);

    if (path.extension() != ".ispc")
      continue;

    std::filesystem::path errorPath(path);

    errorPath.replace_filename(std::string(path.stem().c_str()) +
                               "_stderr.txt");

    auto factory = [path, errorPath]() -> ::testing::Test * {
      return ispc::MakeDiagnosticTest(path.c_str(), errorPath.c_str());
    };

    ::testing::RegisterTest("Diagnostics", path.stem().c_str(), nullptr,
                            nullptr, __FILE__, __LINE__, factory);
  }

  return RUN_ALL_TESTS();
}
