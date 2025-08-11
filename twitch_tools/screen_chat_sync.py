#!/usr/bin/env python3
"""
Система синхронизации чата Twitch с событиями на экране
Создает датасет для обучения нейронки с пониманием контекста
"""
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

class ScreenChatSynchronizer:
    """Синхронизатор чата с событиями на экране"""
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_chat_data(self, chat_file: str) -> List[Dict]:
        """Загружает данные чата из JSON"""
        self.logger.info(f"📂 Загружаю чат из {chat_file}")
        
        with open(chat_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
            
        # Фильтруем только текстовые сообщения
        text_messages = []
        for msg in chat_data:
            if msg.get('message_type') == 'text_message' and msg.get('message'):
                text_messages.append({
                    'time_seconds': msg['time_in_seconds'],
                    'time_text': msg['time_text'],
                    'user': msg['author']['display_name'],
                    'message': msg['message'],
                    'user_badges': [badge['name'] for badge in msg['author'].get('badges', [])],
                    'timestamp': msg['timestamp']
                })
        
        self.logger.info(f"✅ Загружено {len(text_messages)} текстовых сообщений")
        return sorted(text_messages, key=lambda x: x['time_seconds'])
    
    def extract_video_frames(self, video_path: str, output_dir: str, fps: float = 1.0) -> List[Dict]:
        """
        Извлекает кадры из видео с заданным FPS
        
        Args:
            video_path: Путь к видео файлу
            output_dir: Папка для сохранения кадров
            fps: Частота извлечения кадров (1.0 = каждую секунду)
        """
        self.logger.info(f"🎬 Извлекаю кадры из {video_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        # Получаем параметры видео
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        self.logger.info(f"📊 Видео: {video_fps:.1f}fps, {duration:.1f}с, {total_frames} кадров")
        
        # Вычисляем шаг между кадрами
        frame_step = int(video_fps / fps)
        
        extracted_frames = []
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_num % frame_step == 0:
                time_seconds = frame_num / video_fps
                
                # Сохраняем кадр
                frame_filename = f"frame_{int(time_seconds):06d}s.jpg"
                frame_path = output_path / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                extracted_frames.append({
                    'time_seconds': time_seconds,
                    'frame_path': str(frame_path),
                    'frame_number': frame_num
                })
                
                if len(extracted_frames) % 100 == 0:
                    self.logger.info(f"⏳ Извлечено {len(extracted_frames)} кадров...")
            
            frame_num += 1
        
        cap.release()
        self.logger.info(f"✅ Извлечено {len(extracted_frames)} кадров")
        return extracted_frames
    
    def detect_scene_changes(self, frames: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """
        Определяет смену сцен на основе разности кадров
        
        Args:
            frames: Список кадров с путями
            threshold: Порог для определения смены сцены (0.0-1.0)
        """
        self.logger.info(f"🔍 Анализирую смену сцен (порог: {threshold})")
        
        scene_changes = []
        prev_frame = None
        
        for i, frame_info in enumerate(frames):
            current_frame = cv2.imread(frame_info['frame_path'], cv2.IMREAD_GRAYSCALE)
            if current_frame is None:
                continue
                
            if prev_frame is not None:
                # Вычисляем разность кадров
                diff = cv2.absdiff(prev_frame, current_frame)
                diff_normalized = np.mean(diff) / 255.0
                
                if diff_normalized > threshold:
                    scene_changes.append({
                        'time_seconds': frame_info['time_seconds'],
                        'frame_path': frame_info['frame_path'],
                        'change_intensity': diff_normalized,
                        'type': 'scene_change'
                    })
            
            prev_frame = current_frame
        
        self.logger.info(f"✅ Найдено {len(scene_changes)} смен сцен")
        return scene_changes
    
    def group_chat_by_events(self, chat_messages: List[Dict], events: List[Dict], 
                           window_before: int = 10, window_after: int = 30) -> List[Dict]:
        """
        Группирует сообщения чата вокруг событий на экране
        
        Args:
            chat_messages: Сообщения чата
            events: События на экране (смены сцен)
            window_before: Секунд до события для анализа
            window_after: Секунд после события для анализа
        """
        self.logger.info(f"🔄 Группирую чат вокруг {len(events)} событий")
        
        grouped_data = []
        
        for event in events:
            event_time = event['time_seconds']
            
            # Находим сообщения в окне события
            relevant_messages = []
            for msg in chat_messages:
                msg_time = msg['time_seconds']
                if (event_time - window_before) <= msg_time <= (event_time + window_after):
                    # Добавляем относительное время от события
                    msg_copy = msg.copy()
                    msg_copy['relative_time'] = msg_time - event_time
                    relevant_messages.append(msg_copy)
            
            if relevant_messages:  # Только если есть сообщения
                grouped_data.append({
                    'event': event,
                    'messages': relevant_messages,
                    'messages_before': [m for m in relevant_messages if m['relative_time'] < 0],
                    'messages_after': [m for m in relevant_messages if m['relative_time'] >= 0],
                    'total_messages': len(relevant_messages),
                    'analysis_window': {'before': window_before, 'after': window_after}
                })
        
        self.logger.info(f"✅ Создано {len(grouped_data)} групп событие-чат")
        return grouped_data
    
    def analyze_reaction_patterns(self, grouped_data: List[Dict]) -> Dict:
        """Анализирует паттерны реакций в чате"""
        self.logger.info("📊 Анализирую паттерны реакций")
        
        analysis = {
            'total_events': len(grouped_data),
            'average_messages_per_event': 0,
            'common_reaction_words': {},
            'reaction_timing': {'immediate': 0, 'delayed': 0},
            'user_participation': {}
        }
        
        all_words = []
        total_messages = 0
        
        for group in grouped_data:
            total_messages += len(group['messages'])
            
            # Анализируем слова в реакциях
            for msg in group['messages_after']:  # Только реакции после события
                words = msg['message'].lower().split()
                all_words.extend(words)
                
                # Анализируем время реакции
                if abs(msg['relative_time']) <= 5:  # В течение 5 секунд
                    analysis['reaction_timing']['immediate'] += 1
                else:
                    analysis['reaction_timing']['delayed'] += 1
                
                # Анализируем участие пользователей
                user = msg['user']
                if user not in analysis['user_participation']:
                    analysis['user_participation'][user] = 0
                analysis['user_participation'][user] += 1
        
        # Подсчитываем частоту слов
        from collections import Counter
        word_counts = Counter(all_words)
        analysis['common_reaction_words'] = dict(word_counts.most_common(20))
        analysis['average_messages_per_event'] = total_messages / len(grouped_data) if grouped_data else 0
        
        return analysis
    
    def save_training_dataset(self, grouped_data: List[Dict], analysis: Dict, output_file: str):
        """Сохраняет датасет для обучения"""
        self.logger.info(f"💾 Сохраняю датасет в {output_file}")
        
        dataset = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_events': len(grouped_data),
                'description': 'Синхронизированные данные чата и событий на экране для обучения нейронки'
            },
            'analysis': analysis,
            'training_data': grouped_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        self.logger.info("✅ Датасет сохранен!")

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Синхронизация чата Twitch с событиями на экране',
        epilog="""
Примеры использования:
  python screen_chat_sync.py --chat chat.json --video stream.mp4 --output dataset.json
  python screen_chat_sync.py --chat chat.json --video stream.mp4 --fps 0.5 --threshold 0.4
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--chat', required=True, help='Файл с чатом в JSON формате')
    parser.add_argument('--video', required=True, help='Видео файл стрима')
    parser.add_argument('--output', default='training_dataset.json', help='Выходной файл датасета')
    parser.add_argument('--fps', type=float, default=1.0, help='Частота извлечения кадров (по умолчанию: 1.0)')
    parser.add_argument('--threshold', type=float, default=0.3, help='Порог смены сцены (по умолчанию: 0.3)')
    parser.add_argument('--window-before', type=int, default=10, help='Секунд до события (по умолчанию: 10)')
    parser.add_argument('--window-after', type=int, default=30, help='Секунд после события (по умолчанию: 30)')
    parser.add_argument('--frames-dir', default='frames', help='Папка для кадров (по умолчанию: frames)')
    
    args = parser.parse_args()
    
    print("🎬 Синхронизатор чата со событиями на экране")
    print("=" * 50)
    
    try:
        synchronizer = ScreenChatSynchronizer()
        
        # 1. Загружаем чат
        chat_messages = synchronizer.load_chat_data(args.chat)
        
        # 2. Извлекаем кадры из видео
        frames = synchronizer.extract_video_frames(args.video, args.frames_dir, args.fps)
        
        # 3. Определяем смены сцен
        scene_changes = synchronizer.detect_scene_changes(frames, args.threshold)
        
        # 4. Группируем чат вокруг событий
        grouped_data = synchronizer.group_chat_by_events(
            chat_messages, scene_changes, 
            args.window_before, args.window_after
        )
        
        # 5. Анализируем паттерны
        analysis = synchronizer.analyze_reaction_patterns(grouped_data)
        
        # 6. Сохраняем датасет
        synchronizer.save_training_dataset(grouped_data, analysis, args.output)
        
        # Выводим статистику
        print(f"\n📊 Статистика:")
        print(f"   События на экране: {len(scene_changes)}")
        print(f"   Сообщений в чате: {len(chat_messages)}")
        print(f"   События с реакциями: {len(grouped_data)}")
        print(f"   Среднее сообщений на событие: {analysis['average_messages_per_event']:.1f}")
        print(f"   Быстрые реакции: {analysis['reaction_timing']['immediate']}")
        print(f"   Отложенные реакции: {analysis['reaction_timing']['delayed']}")
        
        print(f"\n🎉 Датасет создан: {args.output}")
        
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
