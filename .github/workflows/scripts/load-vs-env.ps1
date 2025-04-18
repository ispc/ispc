if($args[0] -ceq "x86") {
    $arch="x86"
} elseif($args[0] -ceq "x86-64") {
    $arch="x64"
} elseif($args[0] -ceq "aarch64") {
    $arch="arm64"
} else {
	Write-Host "Unrecognized architecture - use of these: x86, x86-64."
	Exit 1
}

# Find Visual Studio installation path
# Note: On ARM Windows, VS installer might be in different location
$vsWherePath = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
if (!(Test-Path $vsWherePath)) {
    $vsWherePath = "${env:ProgramFiles}/Microsoft Visual Studio/Installer/vswhere.exe"
}

${VS_INST_PATH} = & "$vsWherePath" -latest -property installationPath
Write-Output "  <-> VS Install Path: ${VS_INST_PATH}"

# Import DevShell module and enter the dev shell environment
Import-Module ${VS_INST_PATH}/Common7/Tools/Microsoft.VisualStudio.DevShell.dll
Enter-VsDevShell -VsInstallPath ${VS_INST_PATH} -SkipAutomaticLocation -DevCmdArguments "-arch=${arch} -no_logo"
