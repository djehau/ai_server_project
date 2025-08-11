"""
Тест системы долговременной памяти
"""

import requests
import json
import time

def test_memory_service():
    """Тестирует работу сервиса памяти"""
    
    base_url = "http://127.0.0.1:8086"
    test_discord_id = "123456789012345678"
    
    print("🧪 Тестирование системы памяти...")
    
    # 1. Проверка health endpoint
    print("\n1. Проверка health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check: OK")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Не удалось подключиться к сервису памяти: {e}")
        return False
    
    # 2. Сохранение сообщения
    print("\n2. Сохранение сообщения...")
    try:
        response = requests.post(f"{base_url}/memory/save", json={
            "discord_id": test_discord_id,
            "role": "user",
            "message": "Привет, как дела?",
            "emotion": "friendly"
        })
        if response.status_code == 200:
            print("✅ Сообщение сохранено")
        else:
            print(f"❌ Ошибка сохранения: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка при сохранении: {e}")
    
    # 3. Сохранение ответа
    print("\n3. Сохранение ответа...")
    try:
        response = requests.post(f"{base_url}/memory/save", json={
            "discord_id": test_discord_id,
            "role": "vtuber",
            "message": "Привет! Все отлично, спасибо! А как у тебя дела?",
            "emotion": "happy"
        })
        if response.status_code == 200:
            print("✅ Ответ сохранен")
        else:
            print(f"❌ Ошибка сохранения ответа: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка при сохранении ответа: {e}")
    
    # 4. Получение истории
    print("\n4. Получение истории...")
    try:
        response = requests.get(f"{base_url}/memory/history/{test_discord_id}")
        if response.status_code == 200:
            history = response.json()
            print(f"✅ Получена история: {len(history['history'])} сообщений")
            for msg in history['history']:
                print(f"  {msg['role']}: {msg['message']}")
        else:
            print(f"❌ Ошибка получения истории: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка при получении истории: {e}")
    
    # 5. Получение профиля
    print("\n5. Получение профиля...")
    try:
        response = requests.get(f"{base_url}/memory/profile/{test_discord_id}")
        if response.status_code == 200:
            profile = response.json()
            print(f"✅ Профиль получен:")
            print(f"  Discord ID: {profile['discord_id']}")
            print(f"  Всего сообщений: {profile['total_messages']}")
            print(f"  Первое общение: {profile['first_seen']}")
        else:
            print(f"❌ Ошибка получения профиля: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка при получении профиля: {e}")
    
    # 6. Получение контекста
    print("\n6. Получение контекста для AI...")
    try:
        response = requests.get(f"{base_url}/memory/context/{test_discord_id}", 
                              params={"message": "Расскажи о себе"})
        if response.status_code == 200:
            context_data = response.json()
            print(f"✅ Контекст получен:")
            print(f"  {context_data['context'][:200]}...")
        else:
            print(f"❌ Ошибка получения контекста: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка при получении контекста: {e}")
    
    # 7. Статистика
    print("\n7. Получение статистики...")
    try:
        response = requests.get(f"{base_url}/memory/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Статистика:")
            print(f"  Всего сообщений: {stats['total_messages']}")
            print(f"  Уникальных пользователей: {stats['unique_users']}")
            print(f"  Размер БД: {stats['db_size']} байт")
        else:
            print(f"❌ Ошибка получения статистики: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка при получении статистики: {e}")
    
    print("\n🎉 Тест завершен!")
    return True

def test_main_server_chat():
    """Тестирует чат через основной сервер"""
    
    main_url = "http://127.0.0.1:8080"
    test_discord_id = "987654321098765432"
    
    print("\n🗣️ Тестирование чата через основной сервер...")
    
    try:
        # Отправляем сообщение
        response = requests.post(f"{main_url}/chat", json={
            "message": "Привет, помнишь меня?",
            "discord_id": test_discord_id
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Ответ получен: {result.get('response', 'Нет ответа')}")
        else:
            print(f"❌ Ошибка чата: {response.text}")
    
    except Exception as e:
        print(f"❌ Ошибка подключения к основному серверу: {e}")

if __name__ == "__main__":
    # Тестируем сервис памяти
    test_memory_service()
    
    # Небольшая пауза
    time.sleep(2)
    
    # Тестируем чат через основной сервер
    test_main_server_chat()
