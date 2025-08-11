"""
Скрипт для добавления тестовых данных в базу памяти
"""
import sys
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta

# Добавляем путь к сервисам
sys.path.append(str(Path(__file__).parent / "services"))

from memory_service import MemoryService

def add_test_data():
    """Добавляет тестовые данные в базу памяти"""
    print("🧪 Добавление тестовых данных в базу памяти...")
    
    # Создаем экземпляр сервиса памяти
    memory = MemoryService()
    
    # Тестовые пользователи
    test_users = [
        {
            "discord_id": "user123",
            "messages": [
                {"role": "user", "message": "Привет!", "emotion": None},
                {"role": "vtuber", "message": "Привет! Как дела?", "emotion": "happy"},
                {"role": "user", "message": "Все отлично, спасибо!", "emotion": None},
                {"role": "vtuber", "message": "Рада это слышать!", "emotion": "happy"},
                {"role": "user", "message": "Что ты умеешь?", "emotion": None},
                {"role": "vtuber", "message": "Я могу общаться, помогать и развлекать!", "emotion": "excited"},
            ]
        },
        {
            "discord_id": "user456",
            "messages": [
                {"role": "user", "message": "Как работает ИИ?", "emotion": None},
                {"role": "vtuber", "message": "Это очень интересная тема! ИИ работает на основе нейронных сетей...", "emotion": "thinking"},
                {"role": "user", "message": "Расскажи подробнее", "emotion": None},
                {"role": "vtuber", "message": "Нейронные сети обучаются на больших данных...", "emotion": "thinking"},
            ]
        },
        {
            "discord_id": "user789",
            "messages": [
                {"role": "user", "message": "Мне грустно", "emotion": None},
                {"role": "vtuber", "message": "Понимаю... Хочешь поговорить об этом?", "emotion": "sad"},
                {"role": "user", "message": "Да, было бы здорово", "emotion": None},
                {"role": "vtuber", "message": "Я тут, чтобы выслушать тебя", "emotion": "neutral"},
            ]
        }
    ]
    
    # Добавляем данные для каждого пользователя
    for user_data in test_users:
        discord_id = user_data["discord_id"]
        messages = user_data["messages"]
        
        print(f"📝 Добавляем данные для пользователя {discord_id}...")
        
        # Сохраняем каждое сообщение с временными метками
        for i, msg in enumerate(messages):
            # Делаем временные метки с интервалом в несколько минут
            timestamp = datetime.now() - timedelta(minutes=len(messages) - i)
            
            # Сохраняем сообщение
            memory.save_message(
                discord_id=discord_id,
                role=msg["role"],
                message=msg["message"],
                emotion=msg["emotion"],
                session_id=f"test_session_{discord_id}"
            )
    
    print("✅ Тестовые данные добавлены!")
    
    # Показываем статистику
    stats = memory.get_memory_stats()
    print(f"\n📊 Статистика памяти:")
    print(f"   Всего сообщений: {stats.get('total_messages', 0)}")
    print(f"   Уникальных пользователей: {stats.get('unique_users', 0)}")
    print(f"   Размер БД: {stats.get('db_size', 0)} байт")
    
    # Показываем пользователей
    users = memory.get_all_users()
    print(f"\n👥 Пользователи:")
    for user in users:
        print(f"   - {user['discord_id']}: {user['total_messages']} сообщений")

if __name__ == "__main__":
    add_test_data()
