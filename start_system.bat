@echo off
chcp 65001 > nul
echo 🚀 Запуск системы VTuber AI...
echo.

cd /d "%~dp0"

echo Проверяем Python...
python --version
if errorlevel 1 (
    echo ❌ Python не найден! Убедитесь, что Python установлен и добавлен в PATH.
    pause
    exit /b 1
)

echo.
echo ✅ Запускаем основные сервисы...
python start_essential_services.py

echo.
echo 👋 Система завершена.
pause
