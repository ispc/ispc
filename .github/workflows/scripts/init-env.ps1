# Define widely used environment variables

if (-not $env:SDE_MIRROR_ID) {
  $env:SDE_MIRROR_ID = "859732"
  echo "SDE_MIRROR_ID=$env:SDE_MIRROR_ID" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
if (-not $env:SDE_TAR_NAME) {
  $env:SDE_TAR_NAME = "sde-external-9.58.0-2025-06-16"
  echo "SDE_TAR_NAME=$env:SDE_TAR_NAME" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
if (-not $env:USER_AGENT) {
  $env:USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
  echo "USER_AGENT=$env:USER_AGENT" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
if (-not $env:LLVM_REPO) { $env:LLVM_REPO = "https://github.com/ispc/ispc.dependencies" }
if (-not $env:LLVM_VERSION) { $env:LLVM_VERSION = "21.1" }
if (-not $env:LLVM_TAR) { $env:LLVM_TAR = "llvm-21.1.8-win.vs2022-Release+Asserts-x86.arm.wasm.tar.7z" }
if (-not $env:LLVM_HOME) { $env:LLVM_HOME = "C:\\projects\\llvm" }
if (-not $env:CROSS_TOOLS_GNUWIN32) {
  $env:CROSS_TOOLS_GNUWIN32 = "C:\\projects\\cross\\gnuwin32"
  echo "CROSS_TOOLS_GNUWIN32=$env:CROSS_TOOLS_GNUWIN32" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
if (-not $env:BUILD_TYPE) {
  $env:BUILD_TYPE = "Release"
  echo "BUILD_TYPE=$env:BUILD_TYPE" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}

