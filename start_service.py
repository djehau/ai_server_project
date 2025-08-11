#!/usr/bin/env python3
"""
Быстрый запуск отдельных сервисов VTuber AI
"""
import sys
import argparse
import subprocess
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config.microservices import SERVICES

def start_service(service_name: str, background: bool = False):
    """Запускает отдельный сервис"""
    if service_name not in SERVICES:
        print(f"❌ Неизвестный сервис: {service_name}")
        print(f"Доступные сервисы: {', '.join(SERVICES.keys())}")
        return False
    
    service_config = SERVICES[service_name]
    base_dir = Path(__file__).parent
    script_path = base_dir / service_config["process_name"]
    
    if not script_path.exists():
        print(f"❌ Скрипт не найден: {script_path}")
        return False
    
    # Определяем путь к Python
    venv_path = base_dir / "venvs" / service_name
    python_path = venv_path / "Scripts" / "python.exe"
    
    if not python_path.exists():
        print(f"⚠️  Виртуальная среда не найдена: {venv_path}")
        print("   Используется системный Python")
        python_path = "python"
    
    try:
        print(f"🚀 Запуск сервиса: {service_name}")
        print(f"   Описание: {service_config['description']}")
        print(f"   Порт: {service_config['port']}")
        print(f"   Скрипт: {script_path}")
        print(f"   Python: {python_path}")
        
        if background:
            print("   Режим: фоновый")
            # Запуск в фоне
            process = subprocess.Popen(
                [str(python_path), str(script_path)],
                cwd=str(base_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            print(f"✅ Сервис запущен в фоне (PID: {process.pid})")
            return True
        else:
            print("   Режим: интерактивный")
            print("   Нажмите Ctrl+C для остановки")
            print("-" * 50)
            
            # Интерактивный запуск
            result = subprocess.run(
                [str(python_path), str(script_path)],
                cwd=str(base_dir)
            )
            
            if result.returncode == 0:
                print("✅ Сервис завершен успешно")
            else:
                print(f"❌ Сервис завершен с ошибкой (код: {result.returncode})")
            
            return result.returncode == 0
            
    except KeyboardInterrupt:
        print("\n⚠️  Получен сигнал остановки")
        return True
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        return False

def list_services():
    """Показывает список всех сервисов"""
    print("📋 Доступные сервисы:")
    print()
    
    for name, config in SERVICES.items():
        print(f"  {name}")
        print(f"    Описание: {config['description']}")
        print(f"    Порт: {config['port']}")
        print(f"    Скрипт: {config['process_name']}")
        print()

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Быстрый запуск сервисов VTuber AI")
    parser.add_argument("service", nargs="?", help="Имя сервиса для запуска")
    parser.add_argument("--list", "-l", action="store_true", help="Показать список сервисов")
    parser.add_argument("--background", "-b", action="store_true", help="Запустить в фоновом режиме")
    
    args = parser.parse_args()
    
    if args.list:
        list_services()
        return
    
    if not args.service:
        print("🎯 VTuber AI - Быстрый запуск сервисов")
        print()
        print("Использование:")
        print("  python start_service.py [имя_сервиса] [--background]")
        print("  python start_service.py --list")
        print()
        list_services()
        return
    
    success = start_service(args.service, args.background)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
