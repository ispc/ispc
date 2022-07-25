REM Copyright (c) 2023, Intel Corporation
REM SPDX-License-Identifier: BSD-3-Clause

set DEPS_PIPELINE_ID=%1
set ARTIFACTORY=%ARTIFACTORY_ISPC_URL%/llvm-%LLVM_VER%/

REM When the script called with argument that means that
REM we should install that particular LLVM build.
if defined DEPS_PIPELINE_ID (
    set /p CURRENT_LLVM=<%LLVM_HOME%\%LLVM_VER_WITH_SUFFIX%\latest.%LLVM_VER_WITH_SUFFIX%
    if "%DEPS_PIPELINE_ID%" == "%CURRENT_LLVM%" GOTO :EOF

    call %SCRIPTS_DIR%\download-file.bat %ARTIFACTORY%/%DEPS_PIPELINE_ID% llvm-%LLVM_VER_WITH_SUFFIX%.zip %ARTIFACTORY_ISPC_API_KEY% || goto :error
    unzip -o -q llvm-%LLVM_VER_WITH_SUFFIX%.zip
    del /F /Q llvm-%LLVM_VER_WITH_SUFFIX%.zip
    echo %PIPELINE_ID%>%LLVM_VER_WITH_SUFFIX%\latest.%LLVM_VER_WITH_SUFFIX%
    if exist %LLVM_HOME%\%LLVM_VER_WITH_SUFFIX% (
        rd /s /q %LLVM_HOME%\%LLVM_VER_WITH_SUFFIX%
    )
    xcopy %LLVM_VER_WITH_SUFFIX% %LLVM_HOME%\%LLVM_VER_WITH_SUFFIX% /E /I /H /C /K
    rmdir %LLVM_VER_WITH_SUFFIX% /S /Q
    exit /b 0
)

call %SCRIPTS_DIR%\download-file.bat %ARTIFACTORY% latest.%LLVM_VER_WITH_SUFFIX% %ARTIFACTORY_ISPC_API_KEY% || goto :error
set /p LATEST_LLVM=<latest.%LLVM_VER_WITH_SUFFIX%

IF NOT EXIST %LLVM_HOME%\%LLVM_VER_WITH_SUFFIX% GOTO :get_llvm
set /p CURRENT_LLVM=<%LLVM_HOME%\%LLVM_VER_WITH_SUFFIX%\latest.%LLVM_VER_WITH_SUFFIX%
if "%LATEST_LLVM%" == "%CURRENT_LLVM%" GOTO :EOF

:get_llvm
call %SCRIPTS_DIR%\download-file.bat %ARTIFACTORY%/%LATEST_LLVM% llvm-%LLVM_VER_WITH_SUFFIX%.zip %ARTIFACTORY_ISPC_API_KEY% || goto :error
unzip -o -q llvm-%LLVM_VER_WITH_SUFFIX%.zip
del /F /Q llvm-%LLVM_VER_WITH_SUFFIX%.zip
copy latest.%LLVM_VER_WITH_SUFFIX% %LLVM_VER_WITH_SUFFIX%\
if exist %LLVM_HOME%\%LLVM_VER_WITH_SUFFIX% (
    rd /s /q %LLVM_HOME%\%LLVM_VER_WITH_SUFFIX%
)
xcopy %LLVM_VER_WITH_SUFFIX% %LLVM_HOME%\%LLVM_VER_WITH_SUFFIX% /E /I /H /C /K
rmdir %LLVM_VER_WITH_SUFFIX% /S /Q

goto :EOF

:error
echo Failed - error #%errorlevel%
exit /b %errorlevel%
