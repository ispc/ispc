IF NOT DEFINED SCRIPTS_DIR (
  SET "SCRIPTS_DIR=%GITHUB_WORKSPACE%\.github\workflows\scripts\internal"
)
"%VSINSTALLPATH%\VC\Auxiliary\Build\vcvars64.bat" && %*
