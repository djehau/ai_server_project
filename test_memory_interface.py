"""
Тест-скрипт для проверки работы интерфейса просмотра памяти
Запускает основной сервер и сервис памяти для тестирования
"""

import sys
import os
import subprocess
import time
import threading
import webbrowser
from pathlib import Path

def start_memory_service():
    """Запускает сервис памяти"""
    print("🧠 Запуск сервиса памяти...")
    os.chdir(Path(__file__).parent / "services")
    
    try:
        subprocess.run([sys.executable, "memory_api.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Сервис памяти остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска сервиса памяти: {e}")

def start_main_server():
    """Запускает основной сервер"""
    print("🌐 Запуск основного сервера...")
    os.chdir(Path(__file__).parent)
    
    try:
        subprocess.run([sys.executable, "main_server.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Основной сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска основного сервера: {e}")

def open_memory_viewer():
    """Открывает интерфейс просмотра памяти в браузере"""
    time.sleep(3)  # Даем серверу время запуститься
    print("🌐 Открываем интерфейс просмотра памяти...")
    webbrowser.open("http://127.0.0.1:8080/memory/viewer")

def main():
    """Главная функция"""
    print("🚀 Запуск системы для тестирования интерфейса памяти")
    print("=" * 50)
    
    # Запускаем сервис памяти в отдельном потоке
    memory_thread = threading.Thread(target=start_memory_service, daemon=True)
    memory_thread.start()
    
    # Даем время сервису памяти запуститься
    time.sleep(2)
    
    # Открываем браузер в отдельном потоке
    browser_thread = threading.Thread(target=open_memory_viewer, daemon=True)
    browser_thread.start()
    
    # Запускаем основной сервер (блокирующий)
    try:
        start_main_server()
    except KeyboardInterrupt:
        print("\n🛑 Система остановлена пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("✅ Завершение работы")

if __name__ == "__main__":
    main()
