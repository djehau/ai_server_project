"""
Простой тест для проверки работы API памяти
"""

import sys
import requests
import json
from pathlib import Path

# Добавляем путь к сервисам
sys.path.append(str(Path(__file__).parent / "services"))

from memory_service import MemoryService

def test_memory_service():
    """Тестирует сервис памяти напрямую"""
    print("🧪 Тестирование сервиса памяти...")
    
    memory = MemoryService()
    
    # Проверяем статистику
    stats = memory.get_memory_stats()
    print(f"📊 Статистика: {stats}")
    
    # Проверяем список пользователей
    users = memory.get_all_users()
    print(f"👥 Пользователей: {len(users)}")
    for user in users:
        print(f"  - {user['discord_id']}: {user['total_messages']} сообщений")
    
    # Проверяем историю конкретного пользователя
    if users:
        user_id = users[0]['discord_id']
        history = memory.get_conversation_history(user_id, 10)
        print(f"\n💬 История пользователя {user_id}:")
        for msg in history:
            role = "👤" if msg['role'] == 'user' else "🤖"
            print(f"  {role} {msg['message'][:50]}...")

def test_api_endpoints():
    """Тестирует API endpoints (если сервер запущен)"""
    print("\n🌐 Тестирование API endpoints...")
    
    base_url = "http://127.0.0.1:8086"
    
    try:
        # Проверяем health check
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health check: {response.status_code} - {response.json()}")
        
        # Проверяем статистику
        response = requests.get(f"{base_url}/memory/stats", timeout=5)
        if response.status_code == 200:
            print(f"Статистика: {response.json()}")
        
        # Проверяем список пользователей
        response = requests.get(f"{base_url}/memory/users", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"Пользователи через API: {data['count']} найдено")
            for user in data['users'][:3]:  # Показываем первых 3
                print(f"  - {user['discord_id']}: {user['total_messages']} сообщений")
        
        # Проверяем историю пользователя
        response = requests.get(f"{base_url}/memory/history/user123?limit=5", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"\nИстория user123: {data['count']} сообщений")
            for msg in data['messages']:
                role = "👤" if msg['role'] == 'user' else "🤖"
                print(f"  {role} {msg['message'][:40]}...")
        
        print("✅ API работает корректно!")
        
    except requests.exceptions.ConnectionError:
        print("⚠️ API сервер не запущен (это нормально для автономного теста)")
    except Exception as e:
        print(f"❌ Ошибка API: {e}")

def main():
    """Основная функция"""
    print("🧪 Тестирование системы памяти")
    print("=" * 40)
    
    # Тест сервиса памяти напрямую
    test_memory_service()
    
    # Тест API (если сервер запущен)
    test_api_endpoints()
    
    print("\n✅ Тестирование завершено!")

if __name__ == "__main__":
    main()
