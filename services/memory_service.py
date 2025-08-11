"""
Модуль долговременной памяти для VTuber AI
Сохраняет и загружает историю диалогов по Discord ID
"""

import json
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MemoryService:
    """Сервис управления долговременной памятью"""
    
    def __init__(self, db_path: str = "../memory.db"):
        self.db_path = Path(db_path).resolve()  # Используем общую базу на уровень выше
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Создаем таблицу для хранения истории диалогов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    discord_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    emotion TEXT,
                    context TEXT,
                    session_id TEXT
                )
            ''')
            
            # Создаем таблицу для профилей пользователей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    discord_id TEXT PRIMARY KEY,
                    username TEXT,
                    first_seen DATETIME,
                    last_seen DATETIME,
                    total_messages INTEGER DEFAULT 0,
                    preferences TEXT,
                    notes TEXT
                )
            ''')
            
            # Создаем индексы для быстрого поиска
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_discord_id ON conversations(discord_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ База данных памяти инициализирована")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации базы данных: {e}")
    
    def save_message(self, discord_id: str, role: str, message: str, 
                    emotion: str = None, context: str = None, session_id: str = None):
        """Сохраняет сообщение в память"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO conversations (discord_id, timestamp, role, message, emotion, context, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (discord_id, timestamp, role, message, emotion, context, session_id))
            
            # Обновляем профиль пользователя
            self.update_user_profile(cursor, discord_id)
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Сохранено сообщение от {discord_id}: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения сообщения: {e}")
    
    def update_user_profile(self, cursor, discord_id: str):
        """Обновляет профиль пользователя"""
        try:
            # Проверяем, существует ли профиль
            cursor.execute('SELECT discord_id FROM user_profiles WHERE discord_id = ?', (discord_id,))
            exists = cursor.fetchone()
            
            if exists:
                # Обновляем существующий профиль
                cursor.execute('''
                    UPDATE user_profiles 
                    SET last_seen = ?, total_messages = total_messages + 1
                    WHERE discord_id = ?
                ''', (datetime.now().isoformat(), discord_id))
            else:
                # Создаем новый профиль
                cursor.execute('''
                    INSERT INTO user_profiles (discord_id, first_seen, last_seen, total_messages)
                    VALUES (?, ?, ?, 1)
                ''', (discord_id, datetime.now().isoformat(), datetime.now().isoformat()))
                
        except Exception as e:
            logger.error(f"❌ Ошибка обновления профиля: {e}")
    
    def get_conversation_history(self, discord_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Получает историю диалогов пользователя"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, role, message, emotion, context, session_id
                FROM conversations 
                WHERE discord_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (discord_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    'timestamp': row[0],
                    'role': row[1],
                    'message': row[2],
                    'emotion': row[3],
                    'context': row[4],
                    'session_id': row[5]
                })
            
            # Возвращаем в хронологическом порядке
            return list(reversed(history))
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения истории: {e}")
            return []
    
    def get_user_profile(self, discord_id: str) -> Optional[Dict[str, Any]]:
        """Получает профиль пользователя"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT discord_id, username, first_seen, last_seen, total_messages, preferences, notes
                FROM user_profiles 
                WHERE discord_id = ?
            ''', (discord_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'discord_id': row[0],
                    'username': row[1],
                    'first_seen': row[2],
                    'last_seen': row[3],
                    'total_messages': row[4],
                    'preferences': json.loads(row[5]) if row[5] else {},
                    'notes': row[6]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения профиля: {e}")
            return None
    
    def update_user_preferences(self, discord_id: str, preferences: Dict[str, Any]):
        """Обновляет предпочтения пользователя"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_profiles 
                SET preferences = ?
                WHERE discord_id = ?
            ''', (json.dumps(preferences, ensure_ascii=False), discord_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Обновлены предпочтения для {discord_id}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления предпочтений: {e}")
    
    def get_context_for_ai(self, discord_id: str, current_message: str) -> str:
        """Создает контекст для AI на основе истории"""
        try:
            # Получаем профиль пользователя
            profile = self.get_user_profile(discord_id)
            
            # Получаем последние сообщения
            history = self.get_conversation_history(discord_id, limit=10)
            
            context_parts = []
            
            # Информация о пользователе
            if profile:
                context_parts.append(f"Пользователь: {profile.get('username', 'Неизвестный')}")
                context_parts.append(f"Общалися с: {profile.get('first_seen', 'Недавно')}")
                context_parts.append(f"Всего сообщений: {profile.get('total_messages', 0)}")
                
                if profile.get('preferences'):
                    prefs = profile['preferences']
                    context_parts.append(f"Предпочтения: {prefs}")
            
            # Последние сообщения
            if history:
                context_parts.append("\nПоследние сообщения:")
                for msg in history[-5:]:  # Последние 5 сообщений
                    role = "Пользователь" if msg['role'] == 'user' else "Нейро-чан"
                    context_parts.append(f"{role}: {msg['message']}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания контекста: {e}")
            return f"Пользователь {discord_id} отправил сообщение"
    
    def save_conversation(self, discord_id: str, messages: List[Dict[str, Any]], session_id: str = None):
        """Сохраняет целый диалог"""
        try:
            for msg in messages:
                self.save_message(
                    discord_id=discord_id,
                    role=msg.get('role', 'user'),
                    message=msg.get('message', ''),
                    emotion=msg.get('emotion'),
                    context=msg.get('context'),
                    session_id=session_id
                )
            
            logger.info(f"💾 Сохранен диалог: {len(messages)} сообщений")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения диалога: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Получает статистику памяти"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Общее количество сообщений
            cursor.execute('SELECT COUNT(*) FROM conversations')
            total_messages = cursor.fetchone()[0]
            
            # Количество уникальных пользователей
            cursor.execute('SELECT COUNT(DISTINCT discord_id) FROM conversations')
            unique_users = cursor.fetchone()[0]
            
            # Самые активные пользователи
            cursor.execute('''
                SELECT discord_id, COUNT(*) as msg_count
                FROM conversations 
                GROUP BY discord_id 
                ORDER BY msg_count DESC 
                LIMIT 5
            ''')
            active_users = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_messages': total_messages,
                'unique_users': unique_users,
                'active_users': [{'discord_id': u[0], 'messages': u[1]} for u in active_users],
                'db_size': os.path.getsize(str(self.db_path)) if self.db_path.exists() else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Возвращает всех пользователей с сохраненными диалогами"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('''
                SELECT discord_id, username, last_seen, total_messages
                FROM user_profiles
                ORDER BY last_seen DESC
            ''')

            rows = cursor.fetchall()
            conn.close()

            users = []
            for row in rows:
                users.append({
                    'discord_id': row[0],
                    'username': row[1],
                    'last_seen': row[2],
                    'total_messages': row[3]
                })

            return users
        except Exception as e:
            logger.error(f"❌ Ошибка получения пользователей: {e}")
            return []
    
    def delete_message(self, message_id: int) -> bool:
        """Удаляет сообщение по ID"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM conversations WHERE id = ?', (message_id,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            if deleted:
                logger.info(f"✅ Удалено сообщение ID: {message_id}")
            else:
                logger.warning(f"⚠️ Сообщение ID {message_id} не найдено")
            
            return deleted
            
        except Exception as e:
            logger.error(f"❌ Ошибка удаления сообщения: {e}")
            return False
    
    def edit_message(self, message_id: int, new_message: str, new_emotion: str = None) -> bool:
        """Редактирует сообщение"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            if new_emotion is not None:
                cursor.execute(
                    'UPDATE conversations SET message = ?, emotion = ? WHERE id = ?',
                    (new_message, new_emotion, message_id)
                )
            else:
                cursor.execute(
                    'UPDATE conversations SET message = ? WHERE id = ?',
                    (new_message, message_id)
                )
            
            updated = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            if updated:
                logger.info(f"✅ Обновлено сообщение ID: {message_id}")
            else:
                logger.warning(f"⚠️ Сообщение ID {message_id} не найдено")
            
            return updated
            
        except Exception as e:
            logger.error(f"❌ Ошибка редактирования сообщения: {e}")
            return False
    
    def delete_user_data(self, discord_id: str) -> dict:
        """Удаляет все данные пользователя"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Удаляем сообщения
            cursor.execute('DELETE FROM conversations WHERE discord_id = ?', (discord_id,))
            deleted_messages = cursor.rowcount
            
            # Удаляем профиль
            cursor.execute('DELETE FROM user_profiles WHERE discord_id = ?', (discord_id,))
            deleted_profile = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Удалены данные пользователя {discord_id}: {deleted_messages} сообщений, профиль: {deleted_profile}")
            
            return {
                'deleted_messages': deleted_messages,
                'deleted_profile': deleted_profile,
                'success': deleted_messages > 0 or deleted_profile
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка удаления данных пользователя: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_conversation_with_ids(self, discord_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Получает историю с ID сообщений для редактирования"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, role, message, emotion, context, session_id
                FROM conversations 
                WHERE discord_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (discord_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'role': row[2],
                    'message': row[3],
                    'emotion': row[4],
                    'context': row[5],
                    'session_id': row[6]
                })
            
            # Возвращаем в хронологическом порядке
            return list(reversed(history))
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения истории с ID: {e}")
            return []
    
    def clear_old_messages(self, days_old: int = 30):
        """Очищает старые сообщения"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cutoff_date = datetime.now().replace(day=datetime.now().day - days_old)
            
            cursor.execute('''
                DELETE FROM conversations 
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"🗑️ Удалено {deleted_count} старых сообщений")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки старых сообщений: {e}")
            return 0

# Глобальный экземпляр сервиса памяти
memory_service = MemoryService()

def get_memory_service() -> MemoryService:
    """Получает экземпляр сервиса памяти"""
    return memory_service
