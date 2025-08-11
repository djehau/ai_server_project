"""
Менеджер процессов для VTuber AI микросервисов
"""
import subprocess
import time
import signal
import sys
import os
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Устанавливаем кодировку UTF-8 для Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Устанавливаем кодовую страницу консоли
    try:
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
    except:
        pass

from config.microservices import SERVICES, HEALTH_CHECK_INTERVAL

# Настройка логирования с UTF-8 для файлов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/process_manager.log', encoding='utf-8'),
        logging.StreamHandler()  # Оставляем стандартный StreamHandler
    ]
)

# Дополнительный логгер для ошибок
error_logger = logging.getLogger('errors')
error_handler = logging.FileHandler('logs/errors.log', encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
error_logger.addHandler(error_handler)
error_logger.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class ProcessManager:
    """Менеджер для управления микросервисами"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.base_dir = Path(__file__).parent
        self.running = False
        
    def start_service(self, service_name: str) -> bool:
        """Запускает отдельный сервис"""
        if service_name not in SERVICES:
            logger.error(f"Неизвестный сервис: {service_name}")
            return False
            
        service_config = SERVICES[service_name]
        script_path = self.base_dir / service_config["process_name"]
        
        if not script_path.exists():
            logger.error(f"Скрипт не найден: {script_path}")
            return False
            
        try:
            # Временно используем системный Python для всех сервисов
            # TODO: Позже можно настроить отдельные venv с правильными зависимостями
            python_path = sys.executable  # Используем тот же Python, что запустил process_manager
            
            # Запускаем процесс с логированием stdout и stderr
            logs_dir = self.base_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            with open(logs_dir / f"{service_name}_stdout.log", "w", encoding="utf-8") as out_log, \
                 open(logs_dir / f"{service_name}_stderr.log", "w", encoding="utf-8") as err_log:
                
                # Для memory_service нужно запускать с правильным рабочим каталогом
                if service_name == "memory_service":
                    # Меняем рабочий каталог на services/
                    work_dir = self.base_dir / "services"
                    script_name = "memory_api.py"
                else:
                    work_dir = self.base_dir
                    script_name = str(script_path)
                
                process = subprocess.Popen(
                    [str(python_path), script_name] if service_name == "memory_service" else [str(python_path), str(script_path)],
                    stdout=out_log,
                    stderr=err_log,
                    cwd=str(work_dir),
                    env=os.environ.copy()
                )
            
            self.processes[service_name] = process
            logger.info(f"Запущен сервис {service_name} (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска сервиса {service_name}: {e}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Останавливает сервис"""
        if service_name not in self.processes:
            logger.warning(f"Сервис {service_name} не запущен")
            return False
            
        try:
            process = self.processes[service_name]
            process.terminate()
            
            # Ждем завершения
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"Принудительно завершен сервис {service_name}")
            
            del self.processes[service_name]
            logger.info(f"Остановлен сервис {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка остановки сервиса {service_name}: {e}")
            return False
    
    def restart_service(self, service_name: str) -> bool:
        """Перезапускает сервис"""
        logger.info(f"Перезапуск сервиса {service_name}")
        self.stop_service(service_name)
        time.sleep(2)
        return self.start_service(service_name)
    
    def check_service_health(self, service_name: str) -> bool:
        """Проверяет состояние сервиса"""
        if service_name not in self.processes:
            logger.warning(f"Сервис {service_name} не найден в списке процессов")
            return False
            
        process = self.processes[service_name]
        poll_result = process.poll()
        
        # Логируем ошибки если процесс упал
        if poll_result is not None:
            logger.error(f"Процесс {service_name} завершен с кодом {poll_result}")
            
            # Пробуем прочитать ошибки из файла
            try:
                stderr_file = self.base_dir / "logs" / f"{service_name}_stderr.log"
                if stderr_file.exists():
                    with open(stderr_file, "r", encoding="utf-8") as f:
                        stderr_content = f.read().strip()
                        if stderr_content:
                            logger.error(f"Ошибки в {service_name}: {stderr_content[-500:]}")
            except Exception as e:
                logger.error(f"Не удалось прочитать ошибки {service_name}: {e}")
        
        return poll_result is None
    
    def check_http_health(self, service_name: str) -> bool:
        """Проверяет HTTP доступность сервиса"""
        try:
            service_config = SERVICES[service_name]
            url = f"http://{service_config['host']}:{service_config['port']}/health"
            
            import requests
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"HTTP проверка {service_name} вернула код {response.status_code}")
                return False
                
        except ImportError:
            logger.warning(f"HTTP проверка {service_name}: requests не установлен")
            return False
        except Exception as e:
            if 'requests' in str(e):
                logger.warning(f"HTTP проверка {service_name}: ошибка requests - {e}")
            else:
                logger.warning(f"HTTP проверка {service_name}: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, dict]:
        """Возвращает статус всех сервисов"""
        status = {}
        for service_name in SERVICES:
            is_running = self.check_service_health(service_name)
            pid = self.processes.get(service_name).pid if is_running else None
            
            status[service_name] = {
                "running": is_running,
                "pid": pid,
                "description": SERVICES[service_name]["description"]
            }
            
        return status
    
    def start_all_services(self):
        """Запускает все сервисы"""
        logger.info("Запуск всех сервисов...")
        
        # Запускаем в правильном порядке
        start_order = [
            "monitor_service",  # Сначала мониторинг
            "memory_service",   # Память и долговременное хранение
            "llm_service",      # Потом LLM
            "voice_service",    # Голосовые функции
            "avatar_service",   # Аватар
            "obs_service",      # OBS
            "main_server"       # Основной сервер последним
        ]
        
        for service_name in start_order:
            if self.start_service(service_name):
                time.sleep(3)  # Даем время на запуск
            else:
                logger.error(f"Не удалось запустить {service_name}")
        
        self.running = True
        logger.info("Все сервисы запущены")
        
        # Даем всем сервисам время полностью инициализироваться
        logger.info("Ожидание инициализации сервисов...")
        time.sleep(10)
    
    def stop_all_services(self):
        """Останавливает все сервисы"""
        logger.info("Остановка всех сервисов...")
        self.running = False
        
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
        
        logger.info("Все сервисы остановлены")
    
    def monitor_services(self):
        """Мониторинг и автоматический перезапуск сервисов"""
        while self.running:
            try:
                for service_name in SERVICES:
                    process = self.processes.get(service_name)
                    if process and process.poll() is not None:
                        # Сервис остановился
                        logger.error(f"Сервис {service_name} упал с кодом {process.returncode}. Перезапуск...")
                        
                        # Читаем ошибки из файла
                        try:
                            stderr_file = self.base_dir / "logs" / f"{service_name}_stderr.log"
                            if stderr_file.exists():
                                with open(stderr_file, "r", encoding="utf-8") as err_log:
                                    error_content = err_log.read().strip()
                                    if error_content:
                                        logger.error(f"Ошибки в {service_name}: {error_content[-1000:]}")
                        except Exception as e:
                            logger.error(f"Не удалось прочитать ошибки {service_name}: {e}")
                        
                        self.restart_service(service_name)
                    
                    # Дополнительно проверяем HTTP доступность
                    elif process and process.poll() is None:
                        # Процесс работает, проверяем HTTP
                        self.check_http_health(service_name)
                        
                time.sleep(HEALTH_CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Получен сигнал завершения")
                break
            except Exception as e:
                logger.error(f"Ошибка мониторинга: {e}")
                time.sleep(5)
    
    def create_virtual_environments(self):
        """Создает отдельные виртуальные окружения для каждого сервиса"""
        venvs_dir = self.base_dir / "venvs"
        venvs_dir.mkdir(exist_ok=True)
        
        for service_name in SERVICES:
            venv_path = venvs_dir / service_name
            if not venv_path.exists():
                logger.info(f"Создание виртуального окружения для {service_name}")
                try:
                    subprocess.run([
                        sys.executable, "-m", "venv", str(venv_path)
                    ], check=True)
                    logger.info(f"Виртуальное окружение создано: {venv_path}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Ошибка создания venv для {service_name}: {e}")

def signal_handler(signum, frame):
    """Обработчик сигналов для graceful shutdown"""
    logger.info("Получен сигнал завершения")
    if 'manager' in globals():
        manager.stop_all_services()
    sys.exit(0)

if __name__ == "__main__":
    # Устанавливаем обработчики сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Создаем менеджер
    manager = ProcessManager()
    
    # Создаем виртуальные окружения при первом запуске
    manager.create_virtual_environments()
    
    try:
        # Запускаем все сервисы
        manager.start_all_services()
        
        # Запускаем мониторинг
        manager.monitor_services()
        
    except KeyboardInterrupt:
        logger.info("Получен Ctrl+C, завершение работы...")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        manager.stop_all_services()
