REM %IGC_DRIVER_BASE_DIR% - this variable comes from machine env pointing base directory
set ISPC_DEPS_URL=%1

set GFX_NAME=gfx-driver-installer

REM Get gfx.ver file to extract version of GFX we need to install
call %SCRIPTS_DIR%\download-file.bat %ISPC_DEPS_URL% gfx.ver %ARTIFACTORY_ISPC_API_KEY% || goto :error
set /p GFX_VER=<gfx.ver

REM Replace / with \
set TARGET_DIR=%IGC_DRIVER_BASE_DIR%\%GFX_VER:/=\%

set PATH=%TARGET_DIR%\%GFX_NAME%\Graphics;%TARGET_DIR%\%GFX_NAME%;%PATH%
set VC_SPIRVDLL_DIR=%TARGET_DIR%\%GFX_NAME%\Graphics

fc gfx.ver %TARGET_DIR%\%GFX_NAME%\gfx.ver
if %errorlevel% == 0 goto exists

echo "Going to download gfx %GFX_VER%"
call %SCRIPTS_DIR%\download-file.bat %ARTIFACTORY_ISPC_URL%/win-gfx-driver/%GFX_VER% %GFX_NAME%.zip %ARTIFACTORY_ISPC_API_KEY% || goto :error
REM unzip -q %GFX_NAME%.zip || goto :error
7z x -bd %GFX_NAME%.zip || goto :error

move %TARGET_DIR% %TARGET_DIR%_bak
mkdir %TARGET_DIR%
move %GFX_NAME% %TARGET_DIR% || goto :error

rd /s /q %TARGET_DIR%_bak
del /q %GFX_NAME%.zip

:exists
echo "GFX driver %GFX_VER% is up to date"

REM Install driver
call %TARGET_DIR%\%GFX_NAME%\install-gfx-driver.bat %GITHUB_WORKSPACE%\igxpin.log || goto :error

goto :exit

:error
echo Failed - error #%errorlevel%
exit /b %errorlevel%

:exit
