import whisper
import torch
import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

class StreamTranscriber:
    def __init__(self, model_size='large'):
        """Инициализация транскрибера с указанной моделью Whisper"""
        print(f"Загрузка модели Whisper: {model_size}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size).to(device)
        print(f"Использую устройство: {device}")
        print("Модель загружена!")

    def transcribe_audio(self, audio_path, language=None):
        """Транскрибация аудиофайла"""
        print(f"Начинаю транскрибацию файла: {audio_path}")
        
        # Проверяем существование файла
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
        
        # Выполняем транскрибацию
        result = self.model.transcribe(
            audio_path, 
            language=language,
            word_timestamps=True  # Включаем временные метки для слов
        )
        
        print(f"Транскрибация завершена! Найдено сегментов: {len(result.get('segments', []))}")
        return result

    def save_transcription(self, transcription, output_path):
        """Сохранение результата транскрибации в JSON файл"""
        # Создаем директорию если её нет
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Добавляем метаданные
        transcription['metadata'] = {
            'transcribed_at': datetime.now().isoformat(),
            'whisper_model': self.model.__class__.__name__,
            'total_duration': transcription.get('duration', 0),
            'detected_language': transcription.get('language', 'unknown')
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)
        
        print(f"Результат сохранен в: {output_path}")

    def save_text_only(self, transcription, output_path):
        """Сохранение только текста транскрипции"""
        text_path = output_path.replace('.json', '.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(transcription['text'])
        print(f"Текст сохранен в: {text_path}")

    def print_summary(self, transcription):
        """Вывод краткой информации о транскрипции"""
        print("\n=== СВОДКА ТРАНСКРИБАЦИИ ===")
        print(f"Язык: {transcription.get('language', 'не определен')}")
        print(f"Длительность: {transcription.get('duration', 0):.2f} секунд")
        print(f"Количество сегментов: {len(transcription.get('segments', []))}")
        
        # Показываем первые несколько строк
        text = transcription.get('text', '')
        preview = text[:200] + '...' if len(text) > 200 else text
        print(f"Предварительный просмотр: {preview}")
        print("================================\n")

def main():
    parser = argparse.ArgumentParser(description='Транскрибация аудиостримов с помощью Whisper')
    parser.add_argument('audio_path', help='Путь к аудиофайлу')
    parser.add_argument('--model', '-m', default='large', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Размер модели Whisper (по умолчанию: large)')
    parser.add_argument('--language', '-l', help='Код языка (например, ru, en). Если не указан - автоопределение')
    parser.add_argument('--output', '-o', help='Путь для сохранения результата (по умолчанию: data/)')
    parser.add_argument('--text-only', action='store_true', help='Сохранить также файл с чистым текстом')
    
    args = parser.parse_args()
    
    try:
        # Создаем транскрибер
        transcriber = StreamTranscriber(model_size=args.model)
        
        # Выполняем транскрибацию
        transcription = transcriber.transcribe_audio(args.audio_path, language=args.language)
        
        # Определяем путь для сохранения
        if args.output:
            output_path = args.output
        else:
            filename = Path(args.audio_path).stem + '_transcript.json'
            output_path = os.path.join('data', filename)
        
        # Сохраняем результат
        transcriber.save_transcription(transcription, output_path)
        
        # Опционально сохраняем только текст
        if args.text_only:
            transcriber.save_text_only(transcription, output_path)
        
        # Выводим сводку
        transcriber.print_summary(transcription)
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
