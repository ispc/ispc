$msiexecArgs = @(
    "/i",
    "$pwd\ispc-trunk-windows.msi",
    "/L*V",
    "$pwd\install.log",
    "/qn",
    "INSTALL_ROOT=$pwd"
)
Start-Process -FilePath msiexec -ArgumentList $msiexecArgs -NoNewWindow -Wait
cat install.log
echo "$pwd\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
