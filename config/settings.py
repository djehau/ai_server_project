"""
Основные настройки VTuber AI сервера
"""
import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = ASSETS_DIR / "models"
VOICES_DIR = ASSETS_DIR / "voices"

# Создаем директории если их нет
for dir_path in [ASSETS_DIR, LOGS_DIR, MODELS_DIR, VOICES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Сервер настройки
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
DEBUG_MODE = True

# WebSocket настройки
WEBSOCKET_HOST = "127.0.0.1"
WEBSOCKET_PORT = 8765

# AI модель настройки
MODEL_NAME = "microsoft/DialoGPT-medium"  # Можно заменить на локальную модель
MODEL_PATH = MODELS_DIR / "local_model"
USE_GPU = True
MAX_RESPONSE_LENGTH = 100
TEMPERATURE = 0.7

# Голосовые настройки
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
VOICE_SPEED = 1.0
VOICE_PITCH = 1.0

# Настройки экрана
SCREEN_CAPTURE_FPS = 5
SCREEN_ANALYSIS_INTERVAL = 2.0  # секунды

# VTuber настройки
VTUBE_STUDIO_URL = "ws://localhost:8001"
AVATAR_MODEL_PATH = MODELS_DIR / "avatar.model3.json"
EXPRESSION_UPDATE_RATE = 30  # FPS

# OBS настройки
OBS_WEBSOCKET_URL = "ws://localhost:4444"
OBS_WEBSOCKET_PASSWORD = ""

# Мониторинг
MONITOR_INTERVAL = 1.0  # секунды
MAX_GPU_TEMP = 85  # Celsius
MAX_GPU_USAGE = 95  # процент

# Логирование
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "server.log"
CHAT_LOG_FILE = LOGS_DIR / "chat.log"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"

# Чат настройки
CHAT_HISTORY_SIZE = 50
RESPONSE_DELAY_MIN = 1.0  # секунды
RESPONSE_DELAY_MAX = 3.0  # секунды
ENABLE_CHAT_REACTIONS = True

# Эмоции
EMOTION_CATEGORIES = [
    "neutral", "happy", "sad", "angry", "surprised", 
    "confused", "excited", "sleepy", "thinking"
]
DEFAULT_EMOTION = "neutral"
EMOTION_CHANGE_THRESHOLD = 0.3

# Системные лимиты
MAX_CONCURRENT_REQUESTS = 5
REQUEST_TIMEOUT = 30  # секунды
MAX_MEMORY_USAGE = 80  # процент
