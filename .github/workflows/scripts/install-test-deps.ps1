# Choco-Install is GH Actions wrappers around choco, which does retries
Choco-Install -PackageName wget

# Install ISPC package
$msiexecArgs = @(
    "/i",
    "$pwd\ispc-trunk-windows.msi",
    "/L*V",
    "$pwd\install.log",
    "/qn",
    "INSTALL_ROOT=$pwd"
)
Start-Process -FilePath msiexec -ArgumentList $msiexecArgs -NoNewWindow -Wait
cat install.log
echo "$pwd\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append

# Download and unpack SDE
wget --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://software.intel.com/content/dam/develop/external/us/en/documents/downloads/${env:SDE_TAR_NAME}-win.tar.bz2
7z.exe x -tbzip2 ${env:SDE_TAR_NAME}-win.tar.bz2
7z.exe x -ttar ${env:SDE_TAR_NAME}-win.tar
echo "$pwd\${env:SDE_TAR_NAME}-win" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
