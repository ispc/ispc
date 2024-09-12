# Install-ChocoPackage is GH Actions wrappers around choco, which does retries
Install-ChocoPackage wget

# Install ISPC package
Expand-Archive $pwd\ispc-trunk-windows.zip -DestinationPath $pwd
ls
echo "$pwd\ispc-trunk-windows\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append

# Download and unpack SDE
wget -q -U "${env:USER_AGENT}" --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://downloadmirror.intel.com/${env:SDE_MIRROR_ID}/${env:SDE_TAR_NAME}-win.tar.xz
7z.exe x -txz ${env:SDE_TAR_NAME}-win.tar.xz
7z.exe x -ttar ${env:SDE_TAR_NAME}-win.tar
echo "$pwd\${env:SDE_TAR_NAME}-win" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
