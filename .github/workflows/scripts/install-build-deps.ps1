# Install-ChocoPackage is GH Actions wrappers around choco, which does retries
Install-ChocoPackage winflexbison3
Install-ChocoPackage wget
Install-ChocoPackage 7zip
Install-ChocoPackage cygwin
Install-ChocoPackage cyg-get
Install-ChocoPackage ninja

# Install M4 exec and put it into PATH
cyg-get m4
echo "C:\tools\cygwin\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append

# Download and unpack llvm
if ( !(Test-Path ${env:LLVM_HOME}) ) { mkdir ${env:LLVM_HOME} }
cd ${env:LLVM_HOME}
if ( Test-Path env:LLVM_REPO ) { wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 ${env:LLVM_REPO}/releases/download/llvm-${env:LLVM_VERSION}-ispc-dev/${env:LLVM_TAR} }
7z.exe x -t7z ${env:LLVM_TAR}
7z.exe x -ttar llvm*tar
echo "${env:LLVM_HOME}\bin-${env:LLVM_VERSION}\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append

# Download and unpack gnuwin32
mkdir ${env:CROSS_TOOLS_GNUWIN32}
cd ${env:CROSS_TOOLS_GNUWIN32}
# The following line is needed to enable all TLS versions. The server appears to require different TLS version depending on the specific redirect.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls, [Net.SecurityProtocolType]::Tls11, [Net.SecurityProtocolType]::Tls12
wget --retry-connrefused --waitretry=10 --read-timeout=20 --timeout=15 -t 5 -O libgw32c-0.4-lib.zip 'https://github.com/ispc/ispc.dependencies/releases/download/gnuwin32-mirror/libgw32c-0.4-lib.zip'
7z.exe x libgw32c-0.4-lib.zip
