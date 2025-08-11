"""
Конфигурация микросервисной архитектуры для VTuber AI
"""

# Порты для каждого сервиса
SERVICES = {
    "main_server": {
        "port": 8080,
        "host": "127.0.0.1",
        "process_name": "main_server.py",
        "description": "Основной веб-сервер и координатор"
    },
    "llm_service": {
        "port": 8081,
        "host": "127.0.0.1", 
        "process_name": "services/llm_service.py",
        "description": "LLM модель и обработка текста"
    },
    "voice_service": {
        "port": 8082,
        "host": "127.0.0.1",
        "process_name": "services/voice_service.py", 
        "description": "Распознавание речи и TTS"
    },
    "avatar_service": {
        "port": 8083,
        "host": "127.0.0.1",
        "process_name": "services/avatar_service.py",
        "description": "Управление Live2D аватаром"
    },
    "obs_service": {
        "port": 8084,
        "host": "127.0.0.1",
        "process_name": "services/obs_service.py",
        "description": "Интеграция с OBS Studio"
    },
    "monitor_service": {
        "port": 8085,
        "host": "127.0.0.1",
        "process_name": "services/monitor_service.py",
        "description": "Мониторинг системы и GPU"
    },
    "memory_service": {
        "port": 8086,
        "host": "127.0.0.1",
        "process_name": "services/memory_api.py",
        "description": "Долговременная память и история диалогов"
    }
}

# Таймауты для межсервисного общения
SERVICE_TIMEOUT = 30  # секунды
HEALTH_CHECK_INTERVAL = 30  # секунды

# Настройки очередей сообщений
MESSAGE_QUEUE_SIZE = 100
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # секунды
