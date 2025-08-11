"""
Скрипт для запуска основных сервисов: памяти и основного сервера
"""

import subprocess
import sys
import time
import os
import threading
import webbrowser
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EssentialServicesManager:
    def __init__(self, services_to_start=None):
        self.base_dir = Path(__file__).parent
        self.processes = {}
        self.running = False
        
        # Создаем папку для логов
        (self.base_dir / "logs").mkdir(exist_ok=True)
        
        # Определяем какие сервисы запускать
        if services_to_start is None:
            self.services_to_start = ["memory_service", "main_server"]
        else:
            self.services_to_start = services_to_start
        
        logger.info(f"Сервисы для запуска: {', '.join(self.services_to_start)}")
    
    def start_memory_service(self):
        """Запускает сервис памяти"""
        logger.info("🧠 Запуск сервиса памяти...")
        
        try:
            script_path = self.base_dir / "services" / "memory_api.py"
            
            with open("logs/memory_service_stdout.log", "w", encoding="utf-8") as out_log, \
                 open("logs/memory_service_stderr.log", "w", encoding="utf-8") as err_log:
                
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdout=out_log,
                    stderr=err_log,
                    cwd=str(self.base_dir)
                )
            
            self.processes["memory_service"] = process
            logger.info(f"✅ Сервис памяти запущен (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска сервиса памяти: {e}")
            return False
    
    def start_main_server(self):
        """Запускает основной сервер"""
        logger.info("🌐 Запуск основного сервера...")
        
        try:
            script_path = self.base_dir / "main_server.py"
            
            with open("logs/main_server_stdout.log", "w", encoding="utf-8") as out_log, \
                 open("logs/main_server_stderr.log", "w", encoding="utf-8") as err_log:
                
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdout=out_log,
                    stderr=err_log,
                    cwd=str(self.base_dir)
                )
            
            self.processes["main_server"] = process
            logger.info(f"✅ Основной сервер запущен (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска основного сервера: {e}")
            return False
    
    def check_process_health(self, service_name):
        """Проверяет состояние процесса"""
        if service_name not in self.processes:
            return False
        
        process = self.processes[service_name]
        return process.poll() is None
    
    def stop_service(self, service_name):
        """Останавливает сервис"""
        if service_name not in self.processes:
            return
        
        try:
            process = self.processes[service_name]
            process.terminate()
            
            # Ждем завершения
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"⚠️ Принудительно завершен {service_name}")
            
            del self.processes[service_name]
            logger.info(f"🛑 Остановлен {service_name}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка остановки {service_name}: {e}")
    
    def stop_all_services(self):
        """Останавливает все сервисы"""
        logger.info("🛑 Остановка всех сервисов...")
        self.running = False
        
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
        
        logger.info("✅ Все сервисы остановлены")
    
    def monitor_services(self):
        """Мониторинг сервисов"""
        logger.info("👀 Начинаем мониторинг сервисов...")
        
        while self.running:
            try:
                for service_name in list(self.processes.keys()):
                    if not self.check_process_health(service_name):
                        logger.error(f"❌ Сервис {service_name} упал! Перезапуск...")
                        
                        # Читаем ошибки
                        try:
                            stderr_file = self.base_dir / "logs" / f"{service_name}_stderr.log"
                            if stderr_file.exists():
                                with open(stderr_file, "r", encoding="utf-8") as f:
                                    error_content = f.read().strip()
                                    if error_content:
                                        logger.error(f"Ошибки в {service_name}: {error_content[-300:]}")
                        except:
                            pass
                        
                        # Перезапускаем
                        if service_name == "memory_service":
                            self.start_memory_service()
                        elif service_name == "main_server":
                            self.start_main_server()
                
                time.sleep(5)  # Проверяем каждые 5 секунд
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Ошибка мониторинга: {e}")
                time.sleep(5)
    
    def open_browser(self):
        """Открывает браузер с интерфейсом"""
        time.sleep(5)  # Даем время серверу запуститься
        
        try:
            logger.info("🌐 Открываем браузер...")
            webbrowser.open("http://127.0.0.1:8080")
            time.sleep(2)
            webbrowser.open("http://127.0.0.1:8080/memory/viewer")
        except Exception as e:
            logger.error(f"❌ Ошибка открытия браузера: {e}")
    
    def run(self):
        """Основной метод запуска"""
        logger.info("🚀 Запуск системы VTuber AI")
        logger.info("=" * 50)
        
        try:
            # Запускаем сервисы в порядке
            for service_name in self.services_to_start:
                if service_name == "memory_service":
                    if not self.start_memory_service():
                        logger.error(f"❌ Не удалось запустить {service_name}")
                        return
                elif service_name == "main_server":
                    if not self.start_main_server():
                        logger.error(f"❌ Не удалось запустить {service_name}")
                        self.stop_all_services()
                        return
                else:
                    logger.warning(f"⚠️ Сервис {service_name} пока не поддерживается в этом скрипте")
                
                time.sleep(2)  # Даем время на запуск
            
            self.running = True
            
            # Открываем браузер в отдельном потоке
            browser_thread = threading.Thread(target=self.open_browser, daemon=True)
            browser_thread.start()
            
            # Запускаем мониторинг
            self.monitor_services()
            
        except KeyboardInterrupt:
            logger.info("🛑 Получен сигнал завершения (Ctrl+C)")
        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}")
        finally:
            self.stop_all_services()
            logger.info("👋 Завершение работы")

def main():
    """Главная функция"""
    # Проверяем аргументы командной строки
    services_to_start = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            services_to_start = ["memory_service", "main_server"]
            logger.info("🚀 Запуск только основных сервисов")
        elif sys.argv[1] == "--memory-only":
            services_to_start = ["memory_service"]
            logger.info("🧠 Запуск только сервиса памяти")
        elif sys.argv[1] == "--help":
            print("👋 Скрипт запуска сервисов VTuber AI")
            print("Параметры:")
            print("  --all         Запуск основных сервисов (память + основной сервер)")
            print("  --memory-only Запуск только сервиса памяти")
            print("  --help        Показать эту справку")
            print("")
            print("По умолчанию запускается с --all")
            return
    
    manager = EssentialServicesManager(services_to_start)
    
    try:
        manager.run()
    except Exception as e:
        logger.error(f"❌ Фатальная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
