# Twitch Tools - Работа с Twitch API и чатами

Набор инструментов для работы с Twitch API и попытками получения чатов VOD.

## ⚠️ ВАЖНОЕ ПРЕДУПРЕЖДЕНИЕ О ЧАТЕ VOD

**Официальный Twitch API НЕ предоставляет доступ к историческим сообщениям чата VOD!** 

Это ограничение самого Twitch, а не наших инструментов.

## 🔧 Настройка

### 1. Получение API ключей Twitch

1. Перейдите на https://dev.twitch.tv/console/apps
2. Нажмите "Register Your Application"
3. Заполните форму:
   - Name: любое имя
   - OAuth Redirect URLs: `http://localhost:3000`
   - Category: выберите подходящую
4. Получите Client ID и Client Secret

### 2. Создание файла конфигурации

Скопируйте `.env.example` в `.env` и заполните:

```bash
cp .env.example .env
```

Отредактируйте `.env`:
```
TWITCH_CLIENT_ID=ваш_реальный_client_id
TWITCH_CLIENT_SECRET=ваш_реальный_client_secret
```

## 🚀 Использование

### Официальный API (работает для информации о видео)

```bash
# Информация о видео
python chat_downloader.py --video https://twitch.tv/videos/1234567890

# Список недавних видео пользователя
python chat_downloader.py --user streamer_name --list-videos 10

# Информация о пользователе
python chat_downloader.py --user streamer_name
```

### Неофициальные методы (НЕ работают)

```bash
# Демонстрация проблем неофициальных методов
python chat_downloader_unofficial.py --video 1234567890

# Показать рабочие альтернативы
python chat_downloader_unofficial.py --alternatives
```

## 💡 РАБОЧИЕ АЛЬТЕРНАТИВЫ для чата VOD

### 1️⃣ TwitchDownloaderCLI (РЕКОМЕНДУЕТСЯ)

```bash
# Установка
winget install TwitchDownloaderCLI

# Скачать чат
TwitchDownloaderCLI chatdownload -u https://twitch.tv/videos/ID -o chat.json

# Конвертировать в текст
TwitchDownloaderCLI chatrender -i chat.json -o chat.txt --format text
```

### 2️⃣ yt-dlp

```bash
# Установка
winget install yt-dlp

# Скачать комментарии
yt-dlp --write-comments https://twitch.tv/videos/ID
```

### 3️⃣ chat-downloader (Python)

```bash
# Установка
pip install chat-downloader

# Использование
chat_downloader https://twitch.tv/videos/ID
```

## 📁 Структура файлов

```
twitch_tools/
├── chat_downloader.py          # Официальный API (только инфо о видео)
├── chat_downloader_unofficial.py  # Демонстрация неработающих методов
├── .env.example               # Пример конфигурации
├── .env                       # Ваша конфигурация (создайте сами)
├── output/                    # Папка для результатов
├── logs/                      # Логи работы
└── README.md                  # Этот файл
```

## 🔐 Безопасность

- НЕ публикуйте ваши API ключи
- Файл `.env` добавлен в `.gitignore`
- Используйте только официальные методы

## ❌ Что НЕ работает

- Получение исторического чата через Twitch API
- Scraping чата с web-страниц Twitch
- Старые неофициальные API
- Большинство "хаков" и обходных путей

## ✅ Что работает

- Получение информации о видео
- Получение информации о стримерах
- Список недавних видео
- Метаданные VOD

## 🚨 Ограничения Twitch API

1. **Нет доступа к чату VOD** - это политика Twitch
2. **Rate limits** - ограничения на количество запросов
3. **Требуется регистрация приложения**
4. **Только публичная информация**

## 💬 Альтернативы для получения чата

Если вам действительно нужен чат VOD, используйте:

1. **TwitchDownloaderCLI** - самый надежный
2. **yt-dlp** - универсальный инструмент
3. **Браузерные расширения** - для одноразового использования
4. **Запись чата в реальном времени** - во время стрима

## 🤝 Вклад

Если вы знаете работающие легальные способы получения чата VOD, пожалуйста, поделитесь!
