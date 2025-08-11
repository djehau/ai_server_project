import whisper
import torch
import json
import os
import sys
import argparse
import soundfile as sf
import librosa
import numpy as np
from datetime import datetime
from pathlib import Path

class StreamTranscriberAlt:
    def __init__(self, model_size='large'):
        """Инициализация транскрибера с указанной моделью Whisper"""
        print(f"Загрузка модели Whisper: {model_size}...")
        
        # Проверяем доступность CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Использую устройство: {device}")
        
        self.model = whisper.load_model(model_size, device=device)
        print("Модель загружена!")

    def load_audio_alternative(self, audio_path):
        """Альтернативная загрузка аудио без FFmpeg"""
        try:
            # Пробуем загрузить с помощью soundfile
            audio, sr = sf.read(audio_path)
            
            # Если стерео, конвертируем в моно
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                
            # Ресэмплируем до 16kHz если нужно
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
            # Нормализуем
            audio = audio.astype(np.float32)
            
            return audio
            
        except Exception as e:
            print(f"Ошибка загрузки аудио с soundfile: {e}")
            
            # Fallback на librosa
            try:
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                return audio.astype(np.float32)
            except Exception as e2:
                raise Exception(f"Не удалось загрузить аудио ни одним способом: {e}, {e2}")

    def transcribe_audio(self, audio_path, language=None):
        """Транскрибация аудиофайла"""
        print(f"Начинаю транскрибацию файла: {audio_path}")
        
        # Проверяем существование файла
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
        
        try:
            # Загружаем аудио альтернативным способом
            print("Загружаю аудио...")
            audio = self.load_audio_alternative(audio_path)
            print(f"Аудио загружено успешно. Длительность: {len(audio)/16000:.2f} сек")
            
            # Выполняем транскрибацию
            print("Выполняю транскрибацию...")
            result = self.model.transcribe(
                audio, 
                language=language,
                word_timestamps=True,
                verbose=True  # Показываем прогресс
            )
            
            print(f"Транскрибация завершена! Найдено сегментов: {len(result.get('segments', []))}")
            return result
            
        except Exception as e:
            print(f"Ошибка при транскрибации: {e}")
            raise

    def save_transcription(self, transcription, output_path):
        """Сохранение результата транскрибации в JSON файл"""
        # Создаем директорию если её нет
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Добавляем метаданные
        transcription['metadata'] = {
            'transcribed_at': datetime.now().isoformat(),
            'whisper_model': self.model.__class__.__name__,
            'total_duration': transcription.get('duration', 0),
            'detected_language': transcription.get('language', 'unknown'),
            'used_alternative_loader': True
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
    parser = argparse.ArgumentParser(description='Транскрибация аудиостримов с помощью Whisper (альтернативная версия)')
    parser.add_argument('audio_path', help='Путь к аудиофайлу')
    parser.add_argument('--model', '-m', default='medium', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Размер модели Whisper (по умолчанию: medium)')
    parser.add_argument('--language', '-l', help='Код языка (например, ru, en). Если не указан - автоопределение')
    parser.add_argument('--output', '-o', help='Путь для сохранения результата (по умолчанию: data/)')
    parser.add_argument('--text-only', action='store_true', help='Сохранить также файл с чистым текстом')
    
    args = parser.parse_args()
    
    try:
        # Создаем транскрибер
        transcriber = StreamTranscriberAlt(model_size=args.model)
        
        # Выполняем транскрибацию
        transcription = transcriber.transcribe_audio(args.audio_path, language=args.language)
        
        # Определяем путь для сохранения
        if args.output:
            output_path = args.output
        else:
            filename = Path(args.audio_path).stem + '_transcript_alt.json'
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
