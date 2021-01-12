if($args[0] -ceq "x86") {
	$arch="x86"
} elseif($args[0] -ceq "x86-64") {
	$arch="x64"
} else {
	Write-Host "Unrecognized architecture - use of these: x86, x86-64."
	Exit 1
}
${VS_INST_PATH} = & "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe" -latest -property installationPath
Write-Output "  <-> VS Install Path: ${VS_INST_PATH}"
Import-Module ${VS_INST_PATH}/Common7/Tools/Microsoft.VisualStudio.DevShell.dll
Enter-VsDevShell -VsInstallPath ${VS_INST_PATH} -SkipAutomaticLocation -DevCmdArguments "-arch=${arch} -no_logo"
