# Define widely used environment variables

if (-not $env:SDE_REPO) {
  $env:SDE_REPO = "https://github.com/ispc/ispc.dependencies"
  echo "SDE_REPO=$env:SDE_REPO" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
# SDE used by all jobs. A job can pin another version by pre-setting SDE_TAR_NAME.
if (-not $env:SDE_TAR_NAME) {
  $env:SDE_TAR_NAME = "sde-external-10.13.1-2026-07-28"
  echo "SDE_TAR_NAME=$env:SDE_TAR_NAME" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
if (-not $env:LLVM_REPO) { $env:LLVM_REPO = "https://github.com/ispc/ispc.dependencies" }
if (-not $env:LLVM_VERSION) { $env:LLVM_VERSION = "23.1" }
if (-not $env:LLVM_TAR) { $env:LLVM_TAR = "llvm-23.1.2-win.vs2022-Release+Asserts-x86.arm.wasm.tar.7z" }
if (-not $env:LLVM_HOME) { $env:LLVM_HOME = "C:\\projects\\llvm" }
if (-not $env:BUILD_TYPE) {
  $env:BUILD_TYPE = "Release"
  echo "BUILD_TYPE=$env:BUILD_TYPE" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}

