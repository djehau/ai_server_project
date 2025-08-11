#!/usr/bin/env python3
"""
Twitch Chat Downloader - скачивание чатов VOD с таймингом через Twitch API
"""
import asyncio
import json
import os
import sys
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Подключаем библиотеки
from dotenv import load_dotenv
from twitchAPI.twitch import Twitch
from twitchAPI.helper import first
from twitchAPI.object.api import Video
import aiohttp

# Загружаем переменные окружения
load_dotenv()

class TwitchChatDownloader:
    """Класс для скачивания чатов с Twitch VOD"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.twitch = None
        self.setup_logging()
    
    def setup_logging(self):
        """Настройка логирования"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "chat_downloader.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Инициализация Twitch API"""
        self.logger.info("🔑 Инициализация Twitch API...")
        try:
            self.twitch = await Twitch(self.client_id, self.client_secret)
            self.logger.info("✅ Twitch API инициализирован успешно")
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации API: {e}")
            raise
    
    def extract_video_id(self, url_or_id: str) -> str:
        """Извлекает ID видео из URL или возвращает ID как есть"""
        # Паттерны для различных форматов Twitch URL
        patterns = [
            r'twitch\.tv/videos/(\d+)',
            r'twitch\.tv/\w+/v/(\d+)',
            r'(\d+)$'  # Просто число
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        raise ValueError(f"Не удалось извлечь Video ID из: {url_or_id}")
    
    async def get_video_info(self, video_id: str) -> Optional[Video]:
        """Получает информацию о видео"""
        self.logger.info(f"🔍 Получение информации о видео {video_id}...")
        try:
            video = await first(self.twitch.get_videos(video_ids=[video_id]))
            if video:
                self.logger.info(f"📹 Найдено видео: '{video.title}' ({video.duration})")
                self.logger.info(f"📅 Дата: {video.created_at}")
                self.logger.info(f"👤 Стример: {video.user_name}")
                return video
            else:
                self.logger.error(f"❌ Видео с ID {video_id} не найдено")
                return None
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о видео: {e}")
            return None
    
    async def download_chat_comments(self, video_id: str, output_file: str = None) -> bool:
        """
        Скачивает комментарии чата для VOD
        
        ВАЖНО: Официальное Twitch API НЕ предоставляет доступ к историческим 
        комментариям чата для VOD. Это ограничение самого Twitch.
        
        Этот метод показывает, что можно получить через официальный API.
        """
        self.logger.info(f"💬 Попытка получения чата для видео {video_id}...")
        
        # Получаем информацию о видео
        video = await self.get_video_info(video_id)
        if not video:
            return False
        
        # Создаем структуру данных для сохранения
        chat_data = {
            "video_info": {
                "id": video.id,
                "title": video.title,
                "description": video.description,
                "created_at": video.created_at.isoformat(),
                "duration": video.duration,
                "language": video.language,
                "user_name": video.user_name,
                "user_id": video.user_id,
                "view_count": video.view_count,
                "url": video.url
            },
            "chat_messages": [],
            "metadata": {
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "downloader_version": "1.0.0",
                "api_limitation": "Официальный Twitch API не предоставляет доступ к историческим сообщениям чата VOD"
            }
        }
        
        # ВАЖНОЕ ПРИМЕЧАНИЕ для пользователя
        self.logger.warning("⚠️  ОГРАНИЧЕНИЕ TWITCH API:")
        self.logger.warning("📵 Официальный Twitch API НЕ предоставляет доступ к чату VOD")
        self.logger.warning("💡 Для получения чата VOD используйте:")
        self.logger.warning("   1. TwitchDownloaderCLI")
        self.logger.warning("   2. yt-dlp")
        self.logger.warning("   3. Twitch Chat Downloader (третьи решения)")
        
        # Сохраняем то, что можем
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        if not output_file:
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', video.title[:50])
            output_file = output_dir / f"{video_id}_{safe_title}_info.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 Информация о видео сохранена в: {output_file}")
        return True
    
    async def get_user_info(self, username: str):
        """Получает информацию о пользователе"""
        self.logger.info(f"👤 Получение информации о пользователе {username}...")
        try:
            user = await first(self.twitch.get_users(logins=[username]))
            if user:
                self.logger.info(f"✅ Найден пользователь: {user.display_name} (ID: {user.id})")
                return user
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о пользователе: {e}")
        return None
    
    async def list_recent_videos(self, username: str, limit: int = 10):
        """Получает список недавних видео пользователя"""
        user = await self.get_user_info(username)
        if not user:
            return []
        
        self.logger.info(f"📹 Получение последних {limit} видео пользователя {username}...")
        try:
            videos = []
            async for video in self.twitch.get_videos(user_id=user.id, first=limit):
                videos.append({
                    'id': video.id,
                    'title': video.title,
                    'created_at': video.created_at.isoformat(),
                    'duration': video.duration,
                    'view_count': video.view_count,
                    'url': video.url
                })
            
            self.logger.info(f"✅ Найдено {len(videos)} видео")
            return videos
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения видео: {e}")
            return []
    
    async def close(self):
        """Закрытие соединения"""
        if self.twitch:
            await self.twitch.close()

async def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Twitch Chat Downloader - скачивание информации о VOD',
        epilog="""
Примеры использования:
  python chat_downloader.py --video https://twitch.tv/videos/1234567890
  python chat_downloader.py --video 1234567890 --output my_chat.json
  python chat_downloader.py --user streamer_name --list-videos 5
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--video', '-v', help='URL или ID видео на Twitch')
    parser.add_argument('--user', '-u', help='Имя пользователя для получения информации')
    parser.add_argument('--list-videos', '-l', type=int, help='Количество недавних видео для отображения')
    parser.add_argument('--output', '-o', help='Имя файла для сохранения результата')
    
    args = parser.parse_args()
    
    # Проверяем наличие API ключей
    client_id = os.getenv('TWITCH_CLIENT_ID')
    client_secret = os.getenv('TWITCH_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("❌ ОШИБКА: Не найдены API ключи Twitch")
        print("📝 Создайте файл .env со следующими параметрами:")
        print("   TWITCH_CLIENT_ID=ваш_client_id")
        print("   TWITCH_CLIENT_SECRET=ваш_client_secret")
        print("🔗 Получите ключи на: https://dev.twitch.tv/console/apps")
        return
    
    # Создаем загрузчик
    downloader = TwitchChatDownloader(client_id, client_secret)
    
    try:
        await downloader.initialize()
        
        if args.video:
            # Скачиваем информацию о видео
            video_id = downloader.extract_video_id(args.video)
            await downloader.download_chat_comments(video_id, args.output)
            
        elif args.user:
            if args.list_videos:
                # Показываем список видео
                videos = await downloader.list_recent_videos(args.user, args.list_videos)
                if videos:
                    print(f"\n📹 Последние {len(videos)} видео пользователя {args.user}:")
                    for i, video in enumerate(videos, 1):
                        print(f"{i:2}. [{video['id']}] {video['title']}")
                        print(f"    🕐 {video['created_at']} | ⏱️ {video['duration']} | 👁️ {video['view_count']}")
                        print(f"    🔗 {video['url']}\n")
            else:
                # Просто получаем информацию о пользователе
                await downloader.get_user_info(args.user)
        else:
            print("❓ Укажите --video для скачивания или --user для информации о пользователе")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n⏹️  Прервано пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
    finally:
        await downloader.close()

if __name__ == "__main__":
    print("🎮 Twitch Chat Downloader")
    print("=" * 50)
    asyncio.run(main())
