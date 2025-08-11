@echo off
chcp 65001 >nul
echo Twitch API Tools
echo ==================
echo.

:: Переходим в директорию проекта
cd /d "%~dp0"

:: Переходим в папку twitch_tools
cd twitch_tools

:: Проверяем наличие файла .env
if not exist .env (
    echo ОШИБКА: Не найден файл .env
    echo Скопируйте .env.example в .env и заполните API ключи:
    echo.
    echo    copy .env.example .env
    echo.
    echo Получите ключи на: https://dev.twitch.tv/console/apps
    pause
    exit /b 1
)

:: Активируем виртуальное окружение
echo Активируем Twitch окружение...
call ..\venvs\twitch_service\Scripts\activate.bat

:: Показываем меню
:menu
echo.
echo Выберите действие:
echo 1. Информация о видео
echo 2. Список видео пользователя  
echo 3. Информация о пользователе
echo 4. Показать альтернативы для чата
echo 5. Выход
echo.
set /p choice="Введите номер (1-5): "

if "%choice%"=="1" goto video_info
if "%choice%"=="2" goto user_videos
if "%choice%"=="3" goto user_info
if "%choice%"=="4" goto alternatives
if "%choice%"=="5" goto end
goto menu

:video_info
set /p video_url="Введите URL или ID видео: "
echo.
python chat_downloader.py --video "%video_url%"
echo.
pause
goto menu

:user_videos
set /p username="Введите имя пользователя: "
set /p count="Количество видео (по умолчанию 10): "
if "%count%"=="" set count=10
echo.
python chat_downloader.py --user "%username%" --list-videos %count%
echo.
pause
goto menu

:user_info
set /p username="Введите имя пользователя: "
echo.
python chat_downloader.py --user "%username%"
echo.
pause
goto menu

:alternatives
echo.
python chat_downloader_unofficial.py --alternatives
echo.
pause
goto menu

:end
call deactivate
echo До свидания!
pause
