# Инструменты транскрибации стримов

Этот набор инструментов предназначен для автономной транскрибации аудиофайлов стримов с помощью OpenAI Whisper.

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Убедитесь, что у вас установлен FFmpeg (требуется для Whisper):
```bash
# Windows (с помощью chocolatey)
choco install ffmpeg

# Или скачайте с https://ffmpeg.org/download.html
```

## Использование

### Базовое использование
```bash
python transcribe_stream.py путь/к/аудиофайлу.mp3
```

### Дополнительные параметры

```bash
# Выбор размера модели (tiny, base, small, medium, large)
python transcribe_stream.py файл.mp3 --model medium

# Указание языка для лучшего распознавания
python transcribe_stream.py файл.mp3 --language ru

# Сохранение также текстового файла
python transcribe_stream.py файл.mp3 --text-only

# Указание пути для сохранения результата
python transcribe_stream.py файл.mp3 --output output/мой_результат.json

# Полный пример
python transcribe_stream.py стрим.wav --model large --language ru --text-only --output data/стрим_транскрипт.json
```

## Поддерживаемые форматы

Whisper поддерживает множество аудио форматов, включая:
- MP3
- WAV
- FLAC
- M4A
- OGG
- WMA

## Размеры моделей

| Модель | Размер | VRAM | Скорость | Качество |
|--------|---------|------|----------|----------|
| tiny   | 39 MB   | ~1GB | Очень быстро | Базовое |
| base   | 74 MB   | ~1GB | Быстро | Хорошее |
| small  | 244 MB  | ~2GB | Средне | Хорошее |
| medium | 769 MB  | ~5GB | Медленно | Очень хорошее |
| large  | 1550 MB | ~10GB | Очень медленно | Отличное |

## Структура выходных файлов

### JSON файл содержит:
- `text`: Полный текст транскрипции
- `segments`: Массив сегментов с временными метками
- `language`: Определенный язык
- `duration`: Длительность аудио
- `metadata`: Метаданные о процессе транскрибации

### Пример сегмента:
```json
{
  "id": 0,
  "seek": 0,
  "start": 0.0,
  "end": 5.5,
  "text": " Привет, это тестовое сообщение.",
  "tokens": [50364, 38888, ...],
  "temperature": 0.0,
  "avg_logprob": -0.345,
  "compression_ratio": 1.2,
  "no_speech_prob": 0.01
}
```

## Примеры использования

### Транскрибация стрима на русском языке:
```bash
python transcribe_stream.py мой_стрим.mp3 --language ru --text-only
```

### Быстрая транскрибация для предварительного просмотра:
```bash
python transcribe_stream.py файл.wav --model base
```

### Максимальное качество для важного контента:
```bash
python transcribe_stream.py важный_стрим.flac --model large --language ru --text-only
```

## Папки

- `data/` - Здесь сохраняются результаты транскрибации по умолчанию
- `output/` - Альтернативная папка для результатов

## Системные требования

- Python 3.8+
- Достаточно оперативной памяти для выбранной модели
- FFmpeg для декодирования аудио
- CUDA-совместимая видеокарта (опционально, для ускорения)
