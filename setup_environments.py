"""
Скрипт для создания и настройки отдельных виртуальных окружений для каждого микросервиса
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, cwd=None):
    """Выполняет команду и выводит результат"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка: {command}")
        print(f"   {e.stderr}")
        return False

def setup_service_environment(service_name, requirements_file):
    """Настраивает окружение для конкретного сервиса"""
    print(f"\n🔧 Настройка окружения для {service_name}...")
    
    base_dir = Path(__file__).parent
    venv_dir = base_dir / "venvs" / service_name
    venv_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем виртуальное окружение
    if not (venv_dir / "Scripts" / "python.exe").exists():
        print(f"   Создание виртуального окружения...")
        if not run_command(f'python -m venv "{venv_dir}"'):
            return False
    
    # Путь к Python 3.10 для создания окружения
    python310_path = Path(os.environ['USERPROFILE']) / "AppData" / "Local" / "Programs" / "Python" / "Python310" / "python.exe"
    
    # Создаем виртуальное окружение с Python 3.10
    if not (venv_dir / "Scripts" / "python.exe").exists():
        print(f"   Создание виртуального окружения...")
        if not run_command(f'"{python310_path}" -m venv "{venv_dir}"'):
            return False
    
    # Путь к pip в виртуальном окружении
    pip_path = venv_dir / "Scripts" / "pip.exe"
    
    # Обновляем pip
    print(f"   Обновление pip...")
    run_command(f'"{pip_path}" install --upgrade pip')
    
    # Устанавливаем зависимости
    requirements_path = base_dir / requirements_file
    if requirements_path.exists():
        print(f"   Установка зависимостей из {requirements_file}...")
        if service_name == "llm_service":
            # Для LLM сервиса сначала устанавливаем PyTorch с CUDA
            print(f"   Установка PyTorch с CUDA...")
            run_command(f'"{pip_path}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
        
        run_command(f'"{pip_path}" install -r "{requirements_path}"')
    else:
        print(f"   ⚠️  Файл {requirements_file} не найден")
    
    print(f"✅ Окружение для {service_name} готово!")
    return True

def main():
    """Основная функция"""
    print("🚀 Настройка микросервисной архитектуры VTuber AI")
    print("=" * 50)
    
    # Проверяем наличие Python
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        print(f"Python версия: {result.stdout.strip()}")
    except:
        print("❌ Python не найден!")
        return
    
    # Настройка каждого сервиса
    services = [
        ("llm_service", "requirements_llm.txt"),
        ("voice_service", "requirements_voice.txt"), 
        ("avatar_service", "requirements_avatar.txt"),
        ("monitor_service", "requirements_monitor.txt"),
        ("obs_service", "requirements_obs.txt"),
        ("main_server", "requirements_main.txt")
    ]
    
    success_count = 0
    for service_name, requirements_file in services:
        if setup_service_environment(service_name, requirements_file):
            success_count += 1
    
    print(f"\n🎉 Настройка завершена!")
    print(f"✅ Успешно: {success_count}/{len(services)} сервисов")
    
    if success_count == len(services):
        print("\n🔥 Все окружения готовы! Теперь можно запускать:")
        print("   python process_manager.py")
    else:
        print("\n⚠️  Некоторые окружения не удалось настроить. Проверьте ошибки выше.")

if __name__ == "__main__":
    main()
