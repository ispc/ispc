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
wget -U "${env:USER_AGENT}" --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://downloadmirror.intel.com/${env:SDE_MIRROR_ID}/${env:SDE_TAR_NAME}-win.tar.xz
7z.exe x -txz ${env:SDE_TAR_NAME}-win.tar.xz
7z.exe x -ttar ${env:SDE_TAR_NAME}-win.tar
echo "$pwd\${env:SDE_TAR_NAME}-win" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
