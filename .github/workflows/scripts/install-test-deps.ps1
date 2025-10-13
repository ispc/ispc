# Install-ChocoPackage is GH Actions wrappers around choco, which does retries
Install-ChocoPackage wget

# Source the init-env.ps1 script from the same directory as this script
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $SCRIPT_DIR "init-env.ps1")

# Install nanobind for examples build
pip install nanobind
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Install ISPC package
Expand-Archive $pwd\ispc-trunk-windows.zip -DestinationPath $pwd
ls
echo "$pwd\ispc-trunk-windows\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append

# Download and unpack SDE
if (-not $env:USER_AGENT -or -not $env:SDE_MIRROR_ID) {
    Write-Error "SDE_MIRROR_ID and/or USER_AGENT are not defined, exiting."
    exit 1
}
wget -q -U "${env:USER_AGENT}" --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 https://downloadmirror.intel.com/${env:SDE_MIRROR_ID}/${env:SDE_TAR_NAME}-win.tar.xz
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
7z.exe x -txz ${env:SDE_TAR_NAME}-win.tar.xz
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
7z.exe x -ttar ${env:SDE_TAR_NAME}-win.tar
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
echo "$pwd\${env:SDE_TAR_NAME}-win" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
