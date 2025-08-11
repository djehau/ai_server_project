# 🎬 Система синхронизации чата Twitch с событиями на экране

Комплексная система для создания датасета обучения нейронных сетей, которые понимают связь между событиями на экране и реакциями зрителей в чате.

## 📋 Обзор системы

Система состоит из нескольких компонентов:

1. **`screen_chat_sync.py`** — Базовая синхронизация чата с событиями на экране
2. **`advanced_event_detector.py`** — Продвинутый анализ событий (лица, эмоции, движение, текст)
3. **`neural_reaction_model.py`** — Нейронная модель для предсказания реакций чата

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install opencv-python torch torchvision numpy scikit-learn pathlib json argparse logging
```

### 1. Базовая синхронизация чата и видео

```bash
python screen_chat_sync.py --chat chat.json --video stream.mp4 --output basic_dataset.json
```

**Параметры:**
- `--chat` — JSON файл с чатом (из предыдущих инструментов)
- `--video` — Видео файл стрима
- `--output` — Выходной файл датасета
- `--fps 0.5` — Частота анализа кадров (0.5 = каждые 2 секунды)
- `--threshold 0.3` — Порог для детекции смены сцен
- `--window-before 10` — Секунд до события для анализа чата
- `--window-after 30` — Секунд после события для анализа чата

### 2. Улучшение датасета продвинутым анализом

```bash
python advanced_event_detector.py basic_dataset.json frames enhanced_dataset.json
```

Этот шаг добавляет:
- Детекцию лиц и эмоций
- Анализ текстовых областей
- Детекцию движения и изменений цвета
- Классификацию типов событий

### 3. Обучение нейронной модели

```bash
python neural_reaction_model.py --dataset enhanced_dataset.json --epochs 100 --batch-size 16
```

**Параметры:**
- `--dataset` — Файл с улучшенным датасетом
- `--epochs 100` — Количество эпох обучения
- `--batch-size 16` — Размер батча
- `--learning-rate 0.001` — Скорость обучения
- `--model-path reaction_model.pth` — Путь для сохранения модели

## 📊 Структура данных

### Входные данные (чат)

Ожидается JSON с полноформатными сообщениями:

```json
[
  {
    "time_in_seconds": 123.45,
    "time_text": "2:03",
    "message_type": "text_message",
    "message": "POGGERS!",
    "author": {
      "display_name": "ViewerName",
      "badges": [{"name": "subscriber"}]
    },
    "timestamp": "2023-01-01T12:00:00Z"
  }
]
```

### Выходные данные (датасет)

```json
{
  "metadata": {
    "created_at": "2023-01-01T12:00:00",
    "total_events": 150,
    "description": "Синхронизированные данные чата и событий на экране"
  },
  "analysis": {
    "average_messages_per_event": 8.5,
    "reaction_timing": {"immediate": 120, "delayed": 30},
    "common_reaction_words": {"poggers": 15, "wow": 12}
  },
  "training_data": [
    {
      "event": {
        "time_seconds": 123.45,
        "frame_path": "frames/frame_000123s.jpg",
        "change_intensity": 0.75,
        "type": "scene_change"
      },
      "messages": [...],
      "messages_before": [...],
      "messages_after": [...],
      "advanced_analysis": {
        "faces": [...],
        "event_types": ["face_detected", "positive_emotion"],
        "interest_score": 0.85
      }
    }
  ]
}
```

## 🔧 Настройка параметров

### Детекция событий

- **`--threshold`** (0.1-1.0): Чувствительность детекции смены сцен
  - 0.1 — очень чувствительная (много событий)
  - 0.5 — умеренная (рекомендуемая)
  - 0.8 — низкая чувствительность (только крупные изменения)

- **`--fps`** (0.1-5.0): Частота анализа кадров
  - 0.5 — каждые 2 секунды (быстро, экономия места)
  - 1.0 — каждую секунду (рекомендуемая)
  - 2.0 — 2 кадра в секунду (детальный анализ)

### Временные окна

- **`--window-before`** (5-30 сек): Анализ чата до события
- **`--window-after`** (10-60 сек): Анализ чата после события

Рекомендуемые настройки для разных типов контента:

**Геймплей (экшен-игры):**
```bash
--fps 1.0 --threshold 0.4 --window-before 5 --window-after 20
```

**Just Chatting / IRL:**
```bash
--fps 0.5 --threshold 0.6 --window-before 10 --window-after 30
```

**Соревновательные игры:**
```bash
--fps 2.0 --threshold 0.3 --window-before 3 --window-after 15
```

## 🧠 Особенности нейронной модели

### Архитектура

Модель использует мультимодальный подход:

1. **Текстовая ветка**: LSTM для обработки сообщений чата
2. **Визуальная ветка**: Анализ кадров (лица, движение, цвета)
3. **Статистическая ветка**: Метрики чата (количество пользователей, время реакции)
4. **Fusion слой**: Объединение всех модальностей

### Выходы модели

- **Интенсивность реакции** (0-1): Насколько сильно чат отреагировал
- **Тип реакции**: positive, negative, questioning, neutral

### Применение обученной модели

```python
from neural_reaction_model import ReactionPredictor
import numpy as np

# Загружаем модель
predictor = ReactionPredictor('reaction_model.pth')
predictor.load_model(vocab_size=5000)  # Размер словаря из обучения

# Предсказываем реакцию
visual_features = np.array([...])  # 20-мерный вектор визуальных признаков
chat_history = ["previous message 1", "previous message 2"]

prediction = predictor.predict_reaction(visual_features, chat_history)
print(f"Интенсивность: {prediction['predicted_intensity']:.2f}")
print(f"Тип: {prediction['predicted_type']}")
```

## 📈 Анализ результатов

### Метрики качества

После обучения модель выдает метрики:
- **MSE Loss** для интенсивности реакции
- **CrossEntropy Loss** для классификации типа
- **Validation Loss** для контроля переобучения

### Статистика датасета

Система автоматически анализирует:
- Распределение типов событий
- Паттерны реакций (быстрые vs отложенные)
- Популярные слова в реакциях
- Активность пользователей

### Визуализация

```python
# Пример анализа обученной модели
import matplotlib.pyplot as plt
import json

with open('enhanced_dataset.json', 'r') as f:
    data = json.load(f)

# Анализ интенсивности реакций по типам событий
event_types = {}
for item in data['training_data']:
    types = item.get('advanced_analysis', {}).get('event_types', [])
    intensity = item.get('reaction_intensity', 0)
    
    for event_type in types:
        if event_type not in event_types:
            event_types[event_type] = []
        event_types[event_type].append(intensity)

# Строим график
fig, ax = plt.subplots(figsize=(12, 6))
for event_type, intensities in event_types.items():
    ax.hist(intensities, alpha=0.7, label=event_type, bins=20)

ax.set_xlabel('Интенсивность реакции')
ax.set_ylabel('Количество событий')
ax.set_title('Распределение интенсивности реакций по типам событий')
ax.legend()
plt.show()
```

## ⚡ Оптимизация производительности

### Для больших видео (>2 часа):

1. **Используйте низкую частоту кадров**: `--fps 0.5`
2. **Повышайте порог детекции**: `--threshold 0.4`
3. **Уменьшайте временные окна**: `--window-after 20`

### Для ускорения обучения:

1. **Фильтрация данных**: Оставляйте только события с реакциями
2. **Меньший словарь**: Ограничивайте количество уникальных слов
3. **Предварительно обученные эмбеддинги**: Используйте Word2Vec или FastText

### Использование GPU:

Модель автоматически использует CUDA если доступен:

```bash
# Проверка доступности GPU
python -c "import torch; print('CUDA доступен:', torch.cuda.is_available())"
```

## 🐛 Решение проблем

### "Не найдены кадры"
- Проверьте правильность пути к видео
- Убедитесь что видео не повреждено
- Попробуйте другой формат (MP4, AVI)

### "Малое количество событий"
- Уменьшите `--threshold` (например, до 0.2)
- Увеличьте `--fps` (до 1.0 или 2.0)
- Проверьте что видео содержит динамичный контент

### "Модель не сходится"
- Уменьшите learning rate (`--learning-rate 0.0001`)
- Увеличьте количество эпох (`--epochs 200`)
- Проверьте качество данных (достаточно ли событий с реакциями)

### "Ошибка памяти при обучении"
- Уменьшите batch size (`--batch-size 8`)
- Используйте меньший словарь (фильтруйте редкие слова)
- Укоротите max_sequence_length в датасете

## 🔮 Дальнейшие улучшения

### Возможные расширения:

1. **Аудио-анализ**: Добавление обработки звука (громкость, тон, музыка)
2. **Семантический анализ**: Использование BERT/GPT для понимания смысла сообщений  
3. **Временные зависимости**: Учет истории предыдущих событий
4. **Многомасштабный анализ**: События разного временного масштаба
5. **Персонализация**: Учет индивидуальных особенностей зрителей
6. **Реал-тайм предсказание**: Онлайн предсказание реакций во время стрима

### Экспериментальные направления:

- **Cross-modal attention**: Внимание между визуальными и текстовыми модальностями
- **Transformer архитектуры**: Замена LSTM на Transformer
- **Самообучение**: Использование неразмеченных данных
- **Metric learning**: Обучение эмбеддингов для похожих событий

## 📞 Поддержка

При возникновении проблем:

1. Проверьте логи выполнения (включено по умолчанию)
2. Убедитесь в корректности форматов входных данных
3. Проверьте достаточности свободного места на диске
4. Для GPU убедитесь в совместимости версий PyTorch и CUDA

Система протестирована на:
- Python 3.8+
- PyTorch 1.12+
- OpenCV 4.5+
- Windows 10/11, Ubuntu 20.04+

---

## 🔄 Полный пример использования

```bash
# 1. Синхронизируем чат с видео
python screen_chat_sync.py \
    --chat downloaded_chat.json \
    --video stream.mp4 \
    --output basic_dataset.json \
    --fps 1.0 \
    --threshold 0.3 \
    --window-before 10 \
    --window-after 30

# 2. Улучшаем анализ событий
python advanced_event_detector.py \
    basic_dataset.json \
    frames \
    enhanced_dataset.json

# 3. Обучаем нейронную модель
python neural_reaction_model.py \
    --dataset enhanced_dataset.json \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --model-path reaction_model.pth

# 4. Анализируем результаты
echo "Результаты:"
echo "- Базовый датасет: basic_dataset.json"
echo "- Улучшенный датасет: enhanced_dataset.json" 
echo "- Обученная модель: reaction_model.pth"
echo "- Кадры видео: frames/"
```

Теперь у вас есть полная система для анализа связи между событиями на экране и реакциями зрителей в чате! 🎉
