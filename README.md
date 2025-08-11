# VTuber AI Project - Твоя собственная Neuro-sama

## Описание проекта
Создание локального VTuber AI с возможностями:
- Live2D аватар с анимациями и эмоциями
- Голосовой диалог и TTS
- Комментирование игр и происходящего на экране
- Интеграция с OBS для стриминга
- Чат-бот функции для взаимодействия с аудиторией
- Веб-интерфейс для мониторинга и управления
- Отслеживания метрик системы

## Требования к системе
- Python 3.9+
- PyTorch с поддержкой CUDA
- 12GB+ VRAM
- Windows 10/11

## Пошаговая установка

### 1. Установка Python
1. Скачать Python 3.11 с https://www.python.org/downloads/
2. При установке обязательно поставить галочку "Add Python to PATH"
3. Перезагрузить командную строку

### 2. Создание виртуального окружения
```bash
cd C:\Users\user\ai_server_project
python -m venv venv
venv\Scripts\activate
```

### 3. Установка зависимостей
```bash
# Основные ML библиотеки
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate

# Веб-сервер и WebSocket
pip install flask flask-socketio

# Аудио обработка
pip install sounddevice soundfile
pip install TTS
pip install whisper-openai

# Обработка изображений и видео
pip install pillow opencv-python
pip install mediapipe

# Системный мониторинг
pip install psutil py3nvml

# VTuber специфичные
pip install websockets
pip install obs-websocket-py
pip install pydub

# Для чата и стриминга
pip install requests
pip install asyncio
```

## Архитектура проекта
```
ai_server_project/
├── README.md
├── requirements.txt
├── main.py                 # Основной сервер
├── config/
│   ├── settings.py
│   └── character_config.py # Настройки личности AI
├── models/
│   ├── llm_handler.py     # Обработка текстовых запросов
│   ├── voice_handler.py   # Распознавание и синтез речи
│   ├── screen_handler.py  # Анализ экрана
│   └── emotion_handler.py # Обработка эмоций
├── vtuber/
│   ├── avatar_controller.py # Управление Live2D аватаром
│   ├── animation_manager.py # Анимации и жесты
│   ├── emotion_system.py    # Система эмоций
│   └── vtube_studio_api.py  # API для VTube Studio
├── streaming/
│   ├── obs_controller.py   # Управление OBS
│   ├── chat_handler.py     # Обработка чата
│   └── stream_manager.py   # Управление стримом
├── monitoring/
│   ├── gpu_monitor.py     # Мониторинг GPU
│   ├── metrics.py         # Сбор метрик
│   └── performance_tracker.py # Трекинг производительности
├── web/
│   ├── templates/
│   │   ├── index.html     # Главная панель
│   │   ├── avatar.html    # Управление аватаром
│   │   └── stream.html    # Стрим панель
│   └── static/
│       ├── style.css
│       ├── script.js
│       └── avatar_viewer.js
├── assets/
│   ├── models/            # Live2D модели
│   ├── voices/            # Голосовые модели
│   ├── expressions/       # Выражения лица
│   └── animations/        # Анимации
└── logs/
    ├── server.log
    ├── chat.log
    └── errors.log
```

## Микросервисная архитектура
Проект разделен на отдельные процессы для избежания конфликтов зависимостей:
- **LLM Service** (порт 8081) - языковая модель
- **Voice Service** (порт 8082) - распознавание речи и TTS
- **Avatar Service** (порт 8083) - управление Live2D аватаром  
- **OBS Service** (порт 8084) - интеграция с OBS Studio
- **Monitor Service** (порт 8085) - мониторинг системы
- **Main Server** (порт 8080) - веб-интерфейс и координация

## Быстрый старт

### 1. Установка Python
1. Скачать Python 3.11 с https://www.python.org/downloads/
2. При установке поставить галочку "Add Python to PATH"
3. Перезагрузить терминал

### 2. Автоматическая настройка
```bash
cd C:\Users\user\ai_server_project
python setup_environments.py
```
Этот скрипт создаст отдельные виртуальные окружения для каждого сервиса.

### 3. Запуск всей системы
```bash
python process_manager.py
```

### 4. Проверка работы
- Веб-интерфейс: http://localhost:8080
- API документация: http://localhost:8081/docs (LLM)
- Статус сервисов: http://localhost:8080/status

## Управление сервисами

### Ручной запуск отдельного сервиса
```bash
# Активируем нужное окружение
venvs\llm_service\Scripts\activate
# Запускаем сервис
python services/llm_service.py
```

### Команды процесс-менеджера
- `Ctrl+C` - остановка всех сервисов
- Автоматический перезапуск упавших сервисов
- Логи в папке `logs/`

## Интеграция с VTube Studio
1. Скачать VTube Studio
2. Включить "Allow external plugins" в настройках
3. Загрузить Live2D модель в VTube Studio
4. Запустить avatar_service - он автоматически подключится

## Интеграция с OBS
1. Установить OBS Studio
2. Установить плагин obs-websocket
3. В OBS: Tools → WebSocket Server Settings
4. Включить сервер на порту 4444
