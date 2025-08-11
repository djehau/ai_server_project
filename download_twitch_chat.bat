@echo off
chcp 65001 >nul
echo РЕАЛЬНО РАБОТАЮЩИЙ Twitch Chat Downloader
echo ==========================================
echo.

:: Переходим в директорию проекта
cd /d "%~dp0"

:: Проверяем наличие виртуального окружения
if not exist "venvs\twitch_service\Scripts\python.exe" (
    echo ОШИБКА: Виртуальное окружение twitch_service не найдено!
    echo Сначала создайте окружение и установите зависимости.
    pause
    exit /b 1
)

:: Проверяем, передан ли ID видео как параметр
if "%~1"=="" (
    echo.
    echo Использование:
    echo   %~nx0 VIDEO_ID [параметры]
    echo.
    echo Примеры:
    echo   %~nx0 1234567890
    echo   %~nx0 1234567890 -o chat.json
    echo   %~nx0 1234567890 --format text -o chat.txt
    echo   %~nx0 https://twitch.tv/videos/1234567890 --start 300 --end 600
    echo.
    set /p video_id="Введите ID видео или URL: "
    if "!video_id!"=="" (
        echo Не указан ID видео!
        pause
        exit /b 1
    )
) else (
    set video_id=%~1
    shift
)

:: Собираем остальные параметры
set params=
:loop
if "%~1"=="" goto execute
set params=%params% %~1
shift
goto loop

:execute
echo.
echo Скачиваю чат для видео: %video_id%
echo Параметры: %params%
echo.

:: Активируем виртуальное окружение и запускаем скачивание
call venvs\twitch_service\Scripts\activate.bat

:: Переходим в папку twitch_tools
cd twitch_tools

:: Запускаем реальный загрузчик чата
python real_chat_downloader.py %video_id% %params%

:: Показываем результат
echo.
if errorlevel 1 (
    echo ОШИБКА: Не удалось скачать чат!
    echo Возможные причины:
    echo - VOD недоступен или приватный
    echo - Чат отключен для этого VOD
    echo - Неправильный ID видео
) else (
    echo УСПЕХ: Чат скачан!
    echo.
    echo Файлы сохранены в папку: %cd%\output\
    if exist "output\*.json" echo - JSON файлы найдены
    if exist "output\*.txt" echo - Текстовые файлы найдены
    if exist "output\*.csv" echo - CSV файлы найдены
)

:: Деактивируем окружение
call deactivate

echo.
pause
