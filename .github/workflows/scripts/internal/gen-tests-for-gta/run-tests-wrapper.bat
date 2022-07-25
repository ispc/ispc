set TEST_DIR=%1
set ARTIFACTORY_BASE_URL=%2

call %SCRIPTS_DIR%\install-gfx-driver.bat %ARTIFACTORY_BASE_URL% || goto :error

cd %GITHUB_WORKSPACE%
REM call %SCRIPTS_DIR%\download-file.bat %ARTIFACTORY_BASE_URL% level-zero.zip %ARTIFACTORY_ISPC_API_KEY% || goto :error
REM unzip level-zero.zip || goto :error
REM set PATH=%GITHUB_WORKSPACE%\build\install\bin;%PATH%

python %GITHUB_WORKSPACE%\.github\workflows\scripts\internal\gen-tests-for-gta\run-tests.py %TEST_DIR% || goto :error

goto :EOF

:error
echo Failed - error #%errorlevel%
exit /b %errorlevel%
