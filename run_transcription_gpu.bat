@echo off
echo 🎙️ VTuber AI - Транскрибация с GPU поддержкой
echo ================================================
echo.

:: Переходим в директорию проекта
cd /d "%~dp0"

:: Активируем специальное окружение для транскрибации
echo 🔧 Активируем GPU окружение для транскрибации...
call venvs\transcription_service\Scripts\activate.bat

:: Запускаем транскрибатор
echo 🚀 Запускаем транскрибатор...
python run_transcription.py %*

:: Деактивируем окружение
call deactivate

pause
