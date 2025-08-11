"""
Сервис мониторинга системы и GPU
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

@app.route('/')
def index():
    return jsonify({
        "service": "Monitor Service",
        "status": "running",
        "description": "Мониторинг системы и GPU",
        "endpoints": {
            "/": "Главная страница сервиса",
            "/health": "Проверка здоровья",
            "/metrics": "Метрики системы (в разработке)"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "monitor_service"})

@app.route('/metrics')
def metrics():
    import psutil
    
    metrics_data = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('C:\\').percent
    }
    
    # Добавляем информацию о GPU если доступно
    try:
        import torch
        if torch.cuda.is_available():
            metrics_data.update({
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_cached_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "gpu_memory_usage_percent": (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
            })
        else:
            metrics_data["gpu_available"] = False
    except ImportError:
        metrics_data["gpu_available"] = False
    
    return jsonify(metrics_data)

class MonitorService:
    """Сервис мониторинга системы"""
    
    def __init__(self):
        self.running = False
        self.port = SERVICES["monitor_service"]["port"]
    
    def start_flask_server(self):
        """Запуск Flask HTTP сервера"""
        app.run(host='127.0.0.1', port=self.port, debug=False)
    
    def start(self):
        """Запуск сервиса"""
        self.running = True
        logger.info(f"Monitor Service запущен на порту {self.port}")
        
        # Запускаем Flask сервер в отдельном потоке
        flask_thread = threading.Thread(target=self.start_flask_server)
        flask_thread.daemon = True
        flask_thread.start()
        
        try:
            while self.running:
                # Здесь будет логика мониторинга
                logger.info("Мониторинг системы...")
                time.sleep(30)  # Проверка каждые 30 секунд
                
        except KeyboardInterrupt:
            logger.info("Получен сигнал завершения")
            self.stop()
    
    def stop(self):
        """Остановка сервиса"""
        self.running = False
        logger.info("Monitor Service остановлен")

if __name__ == "__main__":
    service = MonitorService()
    service.start()
