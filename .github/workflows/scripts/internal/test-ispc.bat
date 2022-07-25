set TESTS_TARGET=%1
set ISPC_OUTPUT=%2
set FAIL_DB_PATH=%3
set ARTIFACTORY_BASE_URL=%4
set CPU_TARGET=%5

IF not "%CPU_TARGET%"=="" (
  set EXTRA_ARGS=--device=%CPU_TARGET%
)

call %SCRIPTS_DIR%\install-gfx-driver.bat %ARTIFACTORY_BASE_URL% || goto :error

cd %GITHUB_WORKSPACE%
call %SCRIPTS_DIR%\download-file.bat %ARTIFACTORY_BASE_URL% level-zero.zip %ARTIFACTORY_ISPC_API_KEY% || goto :error
unzip level-zero.zip || goto :error
set PATH=%GITHUB_WORKSPACE%\artifacts\install\bin;%PATH%

python run_tests.py -u FP -a xe64 -t %TESTS_TARGET% --l0loader=%GITHUB_WORKSPACE%\level-zero --ispc_output=%ISPC_OUTPUT% --fail_db=%FAIL_DB_PATH% --test_time 60 -j 8 %EXTRA_ARGS% || goto :error

rem :check_exising_processes
rem check for any still running processes
rem tasklist | grep ispc || goto :EOF
rem timeout /T 10
rem goto :check_exising_processes

goto :EOF

:error
echo Failed - error #%errorlevel%
exit /b %errorlevel%
