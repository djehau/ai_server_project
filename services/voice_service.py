"""
Сервис распознавания речи и TTS
"""
import time
import logging
import sys
import threading
from pathlib import Path
from flask import Flask, jsonify

# Добавляем корневую папку в path
sys.path.append(str(Path(__file__).parent.parent))

from config.microservices import SERVICES

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask приложение для HTTP API
app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "voice_service"})

class VoiceService:
    """Сервис для работы с голосом"""
    
    def __init__(self):
        self.running = False
        self.port = SERVICES["voice_service"]["port"]
    
    def start_flask_server(self):
        """Запуск Flask HTTP сервера"""
        app.run(host='127.0.0.1', port=self.port, debug=False)
    
    def start(self):
        """Запуск сервиса"""
        self.running = True
        logger.info(f"Voice Service запущен на порту {self.port}")
        
        # Запускаем Flask сервер в отдельном потоке
        flask_thread = threading.Thread(target=self.start_flask_server)
        flask_thread.daemon = True
        flask_thread.start()
        
        try:
            while self.running:
                # Здесь будет логика обработки голоса
                logger.info("Voice Service работает...")
                time.sleep(60)  # Проверка каждую минуту
                
        except KeyboardInterrupt:
            logger.info("Получен сигнал завершения")
            self.stop()
    
    def stop(self):
        """Остановка сервиса"""
        self.running = False
        logger.info("Voice Service остановлен")

if __name__ == "__main__":
    service = VoiceService()
    service.start()
