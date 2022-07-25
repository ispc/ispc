set ARCH=%1
set TARGET=%2
set O_LEVEL=%3
set OUTPUT_DIR=%4
set ARTIFACTORY_BASE_URL=%5

REM call %SCRIPTS_DIR%\install-gfx-driver.bat %ARTIFACTORY_BASE_URL% || goto :error

cd %GITHUB_WORKSPACE%
call %SCRIPTS_DIR%\download-file.bat %ARTIFACTORY_BASE_URL% level-zero.zip %ARTIFACTORY_ISPC_API_KEY% || goto :error
unzip level-zero.zip || goto :error
set PATH=%GITHUB_WORKSPACE%\artifacts\install\bin;%PATH%

python %GITHUB_WORKSPACE%\.github\workflows\scripts\internal\gen-tests-for-gta\generate.py %ARCH% %TARGET% %O_LEVEL% %OUTPUT_DIR% || goto :error

goto :EOF

:error
echo Failed - error #%errorlevel%
exit /b %errorlevel%
