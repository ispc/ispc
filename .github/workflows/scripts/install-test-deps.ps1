# Install-ChocoPackage is GH Actions wrappers around choco, which does retries
Install-ChocoPackage wget

# Source the init-env.ps1 script from the same directory as this script
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $SCRIPT_DIR "init-env.ps1")

# Install nanobind for examples build
pip install nanobind
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Install ISPC package
$zip = Get-ChildItem -Filter "ispc-*-windows.zip" | Select-Object -First 1
Expand-Archive $zip.FullName -DestinationPath $pwd
ls
$ispcDir = Get-ChildItem -Directory -Filter "ispc-*-windows" | Select-Object -First 1
echo "$pwd\$($ispcDir.Name)\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append

# Download and unpack SDE from the ispc.dependencies release mirror.
if (-not $env:SDE_REPO -or -not $env:SDE_TAR_NAME) {
    Write-Error "SDE_REPO and/or SDE_TAR_NAME are not defined, exiting."
    exit 1
}
wget -q --retry-connrefused --waitretry=5 --read-timeout=20 --timeout=15 -t 5 "${env:SDE_REPO}/releases/download/${env:SDE_TAR_NAME}/${env:SDE_TAR_NAME}-win.tar.xz"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
7z.exe x -txz ${env:SDE_TAR_NAME}-win.tar.xz
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# Extract SDE outside the default workspace ($pwd is D:\a\ispc\ispc). SDE 10.8+
# fails to start when its install path contains a single-character folder (the
# "\a\" segment), so unpack it into a path with no single-character component.
# See https://community.intel.com/t5/Intel-ISA-Extensions/SDE-10-8-fails-to-start-depending-on-the-installation-folder/td-p/1746386
$SDE_INSTALL_DIR = "C:\projects\sde"
7z.exe x -ttar ${env:SDE_TAR_NAME}-win.tar -o"$SDE_INSTALL_DIR"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
echo "$SDE_INSTALL_DIR\${env:SDE_TAR_NAME}-win" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
