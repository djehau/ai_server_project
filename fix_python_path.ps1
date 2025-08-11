# Скрипт для добавления Python в PATH
Write-Host "🔧 Настройка Python PATH..." -ForegroundColor Yellow

# Путь к Python
$pythonPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python313"
$scriptsPath = "$env:USERPROFILE\AppData\Local\Programs\Python\Python313\Scripts"

# Проверяем существование Python
if (Test-Path "$pythonPath\python.exe") {
    Write-Host "✅ Python найден: $pythonPath" -ForegroundColor Green
    
    # Получаем текущий PATH пользователя
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    
    # Проверяем, нет ли уже Python в PATH
    if ($currentPath -notlike "*$pythonPath*") {
        Write-Host "➕ Добавляем Python в PATH..." -ForegroundColor Yellow
        
        # Добавляем Python и Scripts в PATH пользователя
        $newPath = $currentPath + ";" + $pythonPath + ";" + $scriptsPath
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        
        Write-Host "✅ Python добавлен в PATH пользователя" -ForegroundColor Green
    } else {
        Write-Host "ℹ️  Python уже есть в PATH" -ForegroundColor Blue
    }
    
    # Обновляем PATH для текущей сессии
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
    
    Write-Host "🔄 PATH обновлен для текущей сессии" -ForegroundColor Green
    
    # Создаем алиас для python в текущей сессии
    Set-Alias -Name python -Value "$pythonPath\python.exe" -Scope Global
    Set-Alias -Name pip -Value "$scriptsPath\pip.exe" -Scope Global
    
    Write-Host "✅ Алиасы созданы для текущей сессии" -ForegroundColor Green
    
    # Проверяем работу
    Write-Host "🧪 Проверка работы Python..." -ForegroundColor Yellow
    & "$pythonPath\python.exe" --version
    
} else {
    Write-Host "❌ Python не найден в $pythonPath" -ForegroundColor Red
    Write-Host "Попробуйте переустановить Python с официального сайта" -ForegroundColor Yellow
}

Write-Host "`n🎉 Настройка завершена! Теперь можно использовать команды python и pip" -ForegroundColor Green
Write-Host "Примечание: После перезапуска терминала PATH будет работать автоматически" -ForegroundColor Blue
