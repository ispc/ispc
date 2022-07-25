REM Copyright (c) 2023, Intel Corporation
REM SPDX-License-Identifier: BSD-3-Clause

set BUILD_EMBARGO=%1
set ARTIFACTORY_BASE_URL=%2
set DEPS_PIPELINE_ID=%3
set LLVM_HOME=C:\llvm
set LLVM_VER=15.0

set disable_assertions=%LLVM_NO_ASSERTIONS%
set signing_required=%SIGNING_REQUIRED%

set LLVM_ASSERT_SUFFIX

if defined disable_assertions (
    set "LLVM_ASSERT_SUFFIX=.noasserts"
)

set LLVM_VER_WITH_SUFFIX=%LLVM_VER%%LLVM_ASSERT_SUFFIX%

if defined DEPS_PIPELINE_ID (
    call %SCRIPTS_DIR%\install-llvm.bat %DEPS_PIPELINE_ID% || goto :error
) else (
    call %SCRIPTS_DIR%\install-llvm.bat || goto :error
)

set ARTIFACTORY_GFX_BASE_URL=%ARTIFACTORY_ISPC_URL%/ispc-deps

cd %GITHUB_WORKSPACE%
call %SCRIPTS_DIR%\download-file.bat %ARTIFACTORY_BASE_URL% vc-intrinsics.zip %ARTIFACTORY_ISPC_API_KEY% || goto :error
call %SCRIPTS_DIR%\download-file.bat %ARTIFACTORY_BASE_URL% spirv-translator.zip %ARTIFACTORY_ISPC_API_KEY% || goto :error
call %SCRIPTS_DIR%\download-file.bat %ARTIFACTORY_BASE_URL% level-zero.zip %ARTIFACTORY_ISPC_API_KEY% || goto :error
mkdir deps
unzip vc-intrinsics.zip && cp -rf vc-intrinsics/* deps/ || goto :error
unzip spirv-translator.zip && cp -rf spirv-translator/* deps/ || goto :error
unzip level-zero.zip || goto :error

REM Needed to locate llvm-config
set PATH=%LLVM_HOME%\%LLVM_VER_WITH_SUFFIX%\bin-%LLVM_VER%\bin;%PATH%

mkdir build
cd build
REM Build ISPC package ready for release
REM CMake variables used:
REM   ISPC_PREPARE_PACKAGE=ON: prepare ISPC package.
REM   ISPC_INCLUDE_TESTS=ON: needed to run lit-tests for package validation. But the tests are not included in the package itself.
REM   XE_ENABLED=ON, XE_DEPS_DIR: enable Xe support and set the path to directory with Xe dependencies (vc-intrinsics, spir-v translator).
REM   ISPC_CROSS=ON, ISPC_GNUWIN32_PATH: enable cross-compilation support and set the path to directory with GNUWin32 utils.
REM   WASM_ENABLED=OFF: turn off WASM support in release package, it doesn't fully supported.
REM   ISPC_INCLUDE_BENCHMARKS=OFF: don't run benchmarks on Windows in GA.
REM   ISPC_INCLUDE_EXAMPLES=OFF: don't build examples during ISPC build because FileTracker can't process long paths in GA.
cmake -Thost=x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DISPC_PREPARE_PACKAGE=ON -DISPC_INCLUDE_TESTS=ON -DXE_ENABLED=ON -DXE_DEPS_DIR=%GITHUB_WORKSPACE%\deps -DISPC_CROSS=ON -DISPC_GNUWIN32_PATH=C:\gnuwin32 -DWASM_ENABLED=OFF -DISPC_INCLUDE_BENCHMARKS=OFF -DISPC_INCLUDE_EXAMPLES=OFF -D__INTEL_EMBARGO__=%BUILD_EMBARGO% -DLLVM_DIR=%LLVM_HOME%\%LLVM_VER_WITH_SUFFIX%\bin-%LLVM_VER%\lib\cmake\llvm -DLEVEL_ZERO_ROOT=%GITHUB_WORKSPACE%\level-zero ../ || goto :error

REM Build ISPC
MSBuild ispc.sln /p:Configuration=Release /p:Platform=x64 /m || goto :error

REM Run lit tests
cmake --build . --target check-all --config Release || goto :error

REM Sign binaries
if defined signing_required (
    signtool.exe sign /fd sha256 /sha1 %SIGNING_HASH% /tr http://timestamp.comodoca.com/rfc3161 /td sha256 bin/Release/ispc.exe || goto :error
    signtool.exe sign /fd sha256 /sha1 %SIGNING_HASH% /tr http://timestamp.comodoca.com/rfc3161 /td sha256 bin/Release/*.dll || goto :error
)

REM Package ISPC
cmake --build . --target PACKAGE --config Release || goto :error

REM Install ISPC
cmake --build . --target INSTALL --config Release || goto :error

goto :EOF

:error
echo Failed - error #%errorlevel%
exit /b %errorlevel%
