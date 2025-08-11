#!/usr/bin/env python3
"""
РЕАЛЬНО РАБОТАЮЩИЙ Twitch Chat Downloader
Использует проверенную библиотеку chat-downloader
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path
import json
import re
from datetime import datetime

class RealTwitchChatDownloader:
    """Реально работающий загрузчик чата Twitch"""
    
    def __init__(self):
        self.chat_downloader_path = self._find_chat_downloader()
        
    def _find_chat_downloader(self):
        """Находим путь к chat_downloader"""
        # Проверяем в текущем виртуальном окружении
        venv_path = Path(__file__).parent.parent / "venvs" / "twitch_service" / "Scripts"
        
        possible_paths = [
            venv_path / "chat_downloader.exe",
            venv_path / "chat_downloader",
            "chat_downloader"  # Если установлен глобально
        ]
        
        for path in possible_paths:
            try:
                if isinstance(path, Path) and path.exists():
                    return str(path)
                elif isinstance(path, str):
                    # Проверяем, что команда доступна
                    result = subprocess.run([path, "--help"], 
                                          capture_output=True, 
                                          timeout=5)
                    if result.returncode == 0:
                        return path
            except:
                continue
                
        raise FileNotFoundError("chat_downloader не найден. Установите: pip install chat-downloader")
    
    def extract_video_id(self, url_or_id):
        """Извлекает ID видео из URL или возвращает как есть"""
        patterns = [
            r'twitch\.tv/videos/(\d+)',
            r'(\d+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        return url_or_id
    
    def download_chat(self, video_url_or_id, output_file=None, start_time=None, end_time=None, format_type="json"):
        """
        Скачивает чат VOD
        
        Args:
            video_url_or_id: URL или ID видео
            output_file: Файл для сохранения
            start_time: Время начала (в секундах или hh:mm:ss)
            end_time: Время окончания (в секундах или hh:mm:ss) 
            format_type: Формат вывода (json, csv, text)
        """
        print(f"🚀 Начинаю скачивание чата...")
        
        # Формируем URL
        video_id = self.extract_video_id(video_url_or_id)
        if not video_url_or_id.startswith("http"):
            video_url = f"https://www.twitch.tv/videos/{video_id}"
        else:
            video_url = video_url_or_id
            
        print(f"📹 Video URL: {video_url}")
        
        # Создаем команду
        cmd = [self.chat_downloader_path, video_url]
        
        # Добавляем параметры
        if output_file:
            cmd.extend(["--output", output_file])
            
        if start_time:
            cmd.extend(["--start_time", str(start_time)])
            
        if end_time:
            cmd.extend(["--end_time", str(end_time)])
            
        # Добавляем формат если не json
        if format_type != "json":
            if format_type == "text":
                cmd.extend(["--format", "{timestamp} {author[display_name]}: {message}"])
            elif format_type == "csv":
                cmd.extend(["--format", "{timestamp},{author[display_name]},{message}"])
        
        # Настройки для лучшей работы
        cmd.extend([
            "--message_groups", "messages",  # Только сообщения
            "--sort_keys",
            "--indent", "2"
        ])
        
        print(f"🔧 Команда: {' '.join(cmd)}")
        print(f"💾 Сохранение в: {output_file or 'консоль'}")
        print("⏳ Скачивание может занять время в зависимости от длины VOD...")
        print()
        
        try:
            # Запускаем загрузку
            result = subprocess.run(cmd, check=True, text=True, 
                                  capture_output=not bool(output_file))
            
            if output_file:
                print(f"✅ Чат успешно сохранен в: {output_file}")
                # Показываем статистику файла
                if Path(output_file).exists():
                    file_size = Path(output_file).stat().st_size
                    print(f"📊 Размер файла: {file_size / 1024:.1f} KB")
                    
                    if output_file.endswith('.json'):
                        try:
                            with open(output_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    print(f"📝 Количество сообщений: {len(data)}")
                        except:
                            pass
            else:
                print("✅ Чат выведен в консоль")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка при скачивании чата: {e}")
            print(f"💡 Возможные причины:")
            print(f"   - VOD недоступен или приватный")
            print(f"   - Чат отключен для этого VOD") 
            print(f"   - Неправильный ID видео")
            return False
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")
            return False
            
        return True

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='РЕАЛЬНО РАБОТАЮЩИЙ Twitch Chat Downloader',
        epilog="""
Примеры использования:
  python real_chat_downloader.py https://twitch.tv/videos/1234567890
  python real_chat_downloader.py 1234567890 -o chat.json
  python real_chat_downloader.py 1234567890 -o chat.txt --format text --start 300 --end 900
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('video', help='URL или ID видео Twitch VOD')
    parser.add_argument('-o', '--output', help='Файл для сохранения чата')
    parser.add_argument('-s', '--start', help='Время начала (в секундах или hh:mm:ss)')
    parser.add_argument('-e', '--end', help='Время окончания (в секундах или hh:mm:ss)')
    parser.add_argument('--format', choices=['json', 'text', 'csv'], 
                       default='json', help='Формат вывода (по умолчанию: json)')
    
    args = parser.parse_args()
    
    print("💬 РЕАЛЬНО РАБОТАЮЩИЙ Twitch Chat Downloader")
    print("=" * 50)
    
    try:
        downloader = RealTwitchChatDownloader()
        
        # Генерируем имя файла если не указано
        if not args.output and args.format in ['json', 'text', 'csv']:
            video_id = downloader.extract_video_id(args.video)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = args.format if args.format != 'text' else 'txt'
            args.output = f"chat_{video_id}_{timestamp}.{ext}"
            print(f"📁 Файл не указан, сохраняю в: {args.output}")
        
        # Создаем папку output если нужно
        if args.output:
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                args.output = str(output_dir / args.output)
        
        # Скачиваем чат
        success = downloader.download_chat(
            args.video,
            args.output, 
            args.start,
            args.end,
            args.format
        )
        
        if success:
            print("\n🎉 Готово! Чат успешно скачан!")
        else:
            print("\n💔 Не удалось скачать чат")
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 Для установки выполните:")
        print("   pip install chat-downloader")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
