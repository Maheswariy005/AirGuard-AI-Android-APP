$env:ANDROID_HOME = "C:\Users\ADMIN\AppData\Local\Android\Sdk"
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21.0.12.1"
$gradle = "C:\Users\ADMIN\.gradle\wrapper\dists\gradle-9.1.0-bin\9agqghryom9wkf8r80qlhnts3\gradle-9.1.0\bin\gradle.bat"
Set-Location "C:\Users\ADMIN\AppData\AirGuardAI"
& $gradle assembleDebug 2>&1 | Out-File "C:\Users\ADMIN\AppData\build_out.txt" -Encoding utf8
Write-Host "EXIT:$LASTEXITCODE"
