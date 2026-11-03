# Install-ChocoPackage is GH Actions wrappers around choco, which does retries
Install-ChocoPackage winflexbison3
Install-ChocoPackage wget
Install-ChocoPackage 7zip
Install-ChocoPackage gnuwin32-m4
Install-ChocoPackage ninja

# Source the init-env.ps1 script from the same directory as this script
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $SCRIPT_DIR "init-env.ps1")

# Download and unpack llvm
if ( !(Test-Path ${env:LLVM_HOME}) ) { mkdir ${env:LLVM_HOME} }
cd ${env:LLVM_HOME}
if ( Test-Path env:LLVM_REPO ) { wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 ${env:LLVM_REPO}/releases/download/llvm-${env:LLVM_VERSION}-ispc-dev/${env:LLVM_TAR}; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
7z.exe x -t7z ${env:LLVM_TAR}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
7z.exe x -ttar llvm*tar
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
echo "${env:LLVM_HOME}\bin-${env:LLVM_VERSION}\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
exit $LASTEXITCODE
