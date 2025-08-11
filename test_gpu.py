#!/usr/bin/env python3
"""
Простой скрипт для проверки работы GPU в проекте VTuber AI
"""
import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))

def test_gpu():
    """Тестирует доступность GPU"""
    print("🔍 Проверка доступности GPU...")
    
    try:
        import torch
        print(f"✅ PyTorch версия: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA доступна: {torch.version.cuda}")
            print(f"✅ Количество GPU: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
            # Тестируем простую операцию на GPU
            print("\n🧪 Тестирование вычислений на GPU...")
            device = torch.device('cuda:0')
            
            # Создаем небольшую матрицу на GPU
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            # Выполняем матричное умножение
            result = torch.mm(x, y)
            print(f"✅ Матричное умножение на GPU выполнено успешно!")
            print(f"   Результат: {result.shape} tensor на {result.device}")
            
            # Проверяем использование памяти
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"   Использование памяти: {memory_allocated:.2f} GB (выделено), {memory_cached:.2f} GB (кэшировано)")
            
        else:
            print("❌ CUDA недоступна - будет использован CPU")
            return False
            
    except ImportError:
        print("❌ PyTorch не установлен")
        return False
        
    return True

def test_llm_service():
    """Тестирует LLM сервис"""
    print("\n🤖 Тестирование LLM сервиса...")
    
    try:
        # Импортируем LLM handler
        sys.path.append(str(Path(__file__).parent / "services"))
        from llm_service import LLMHandler
        
        # Создаем обработчик
        handler = LLMHandler()
        print(f"✅ LLM Handler создан, устройство: {handler.device}")
        
        if handler.device == "cuda":
            print("✅ LLM сервис настроен на использование GPU")
            return True
        else:
            print("⚠️  LLM сервис будет использовать CPU")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании LLM сервиса: {e}")
        return False

def main():
    """Основная функция"""
    print("🎯 VTuber AI - Тест GPU\n")
    
    gpu_ok = test_gpu()
    llm_ok = test_llm_service()
    
    print("\n📊 Результаты тестирования:")
    print(f"   GPU: {'✅ OK' if gpu_ok else '❌ Fail'}")
    print(f"   LLM: {'✅ OK' if llm_ok else '❌ Fail'}")
    
    if gpu_ok and llm_ok:
        print("\n🎉 Все тесты пройдены! Ваш проект готов к работе с GPU.")
    else:
        print("\n⚠️  Некоторые тесты не прошли. Проверьте установку PyTorch с поддержкой CUDA.")
        print("   Для установки используйте:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")

if __name__ == "__main__":
    main()
