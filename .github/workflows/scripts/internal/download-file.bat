rem download-file(url, file_name, api_key)
curl --connect-timeout 5 --max-time 3600 --retry 5 --retry-delay 0 --retry-max-time 40 --fail -H "X-JFrog-Art-Api:%3" "%1/%2" --output %2
exit /b %errorlevel%
