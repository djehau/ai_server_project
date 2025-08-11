#!/usr/bin/env python3
"""
Нейронная модель для анализа синхронизированных данных чата и видео
Предсказывает реакции чата на основе событий на экране
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from typing import List, Dict, Tuple
import pickle
import logging
from pathlib import Path

class ChatReactionDataset(Dataset):
    """Датасет для обучения модели предсказания реакций чата"""
    
    def __init__(self, data: List[Dict], tokenizer=None, max_sequence_length=512):
        self.data = data
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        
        # Создаем простой токенизатор если не передан
        if self.tokenizer is None:
            self.tokenizer = self.create_simple_tokenizer()
        
        # Подготавливаем данные
        self.processed_data = self.preprocess_data()
    
    def create_simple_tokenizer(self):
        """Создает простой токенизатор на основе данных"""
        all_words = set()
        
        for item in self.data:
            messages = item.get('messages', [])
            for msg in messages:
                words = msg['message'].lower().split()
                all_words.update(words)
        
        # Создаем словарь
        word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        for word in sorted(all_words):
            word_to_idx[word] = len(word_to_idx)
        
        return word_to_idx
    
    def tokenize_text(self, text: str) -> List[int]:
        """Токенизирует текст"""
        words = text.lower().split()
        tokens = [self.tokenizer.get(word, 1) for word in words]  # 1 = <UNK>
        
        # Добавляем START и END токены
        tokens = [2] + tokens + [3]  # 2 = <START>, 3 = <END>
        
        # Обрезаем или дополняем до нужной длины
        if len(tokens) > self.max_sequence_length:
            tokens = tokens[:self.max_sequence_length]
        else:
            tokens.extend([0] * (self.max_sequence_length - len(tokens)))  # 0 = <PAD>
        
        return tokens
    
    def preprocess_data(self):
        """Предобрабатывает данные для обучения"""
        processed = []
        
        for item in self.data:
            # Извлекаем визуальные признаки
            visual_features = self.extract_visual_features(item)
            
            # Извлекаем текстовые признаки чата
            chat_features = self.extract_chat_features(item)
            
            # Создаем целевую переменную (интенсивность реакции)
            reaction_intensity = self.calculate_reaction_intensity(item)
            
            # Классификация типа реакции
            reaction_type = self.classify_reaction_type(item)
            
            processed.append({
                'visual_features': visual_features,
                'chat_features': chat_features,
                'reaction_intensity': reaction_intensity,
                'reaction_type': reaction_type,
                'messages_count': len(item.get('messages', [])),
                'event_types': item.get('event', {}).get('type', 'unknown')
            })
        
        return processed
    
    def extract_visual_features(self, item: Dict) -> np.ndarray:
        """Извлекает визуальные признаки из события"""
        features = np.zeros(20)  # 20-мерный вектор визуальных признаков
        
        event = item.get('event', {})
        advanced_analysis = item.get('advanced_analysis', {})
        
        # Базовые признаки
        features[0] = event.get('change_intensity', 0.0)
        features[1] = advanced_analysis.get('interest_score', 0.0)
        features[2] = advanced_analysis.get('total_faces', 0)
        features[3] = advanced_analysis.get('total_text_regions', 0)
        
        # Движение и цвет
        motion_info = advanced_analysis.get('motion_info', {})
        color_info = advanced_analysis.get('color_info', {})
        
        features[4] = motion_info.get('motion_intensity', 0.0)
        features[5] = motion_info.get('motion_ratio', 0.0)
        features[6] = color_info.get('avg_brightness', 0.0) / 255.0  # Нормализация
        features[7] = color_info.get('brightness_change', 0.0) / 100.0  # Нормализация
        
        # Лица и эмоции
        faces = advanced_analysis.get('faces', [])
        if faces:
            features[8] = 1.0  # Есть лица
            features[9] = max(face.get('smile_confidence', 0.0) for face in faces)
            features[10] = sum(face.get('face_area', 0) for face in faces) / 10000.0  # Нормализация
            features[11] = len([face for face in faces if face.get('has_smile', False)]) / max(len(faces), 1)
        
        # Типы событий (one-hot encoding)
        event_types = advanced_analysis.get('event_types', [])
        event_type_mapping = {
            'face_detected': 12,
            'positive_emotion': 13,
            'text_present': 14,
            'high_motion': 15,
            'brightness_change': 16,
            'color_shift': 17,
            'scene_change': 18
        }
        
        for event_type in event_types:
            if event_type in event_type_mapping:
                features[event_type_mapping[event_type]] = 1.0
        
        # Временные признаки
        features[19] = event.get('time_seconds', 0.0) / 3600.0  # Время в часах
        
        return features
    
    def extract_chat_features(self, item: Dict) -> Dict:
        """Извлекает признаки из сообщений чата"""
        messages = item.get('messages', [])
        messages_before = item.get('messages_before', [])
        messages_after = item.get('messages_after', [])
        
        # Токенизируем все сообщения
        all_messages_text = ' '.join([msg['message'] for msg in messages])
        tokenized_messages = self.tokenize_text(all_messages_text)
        
        # Статистики чата
        unique_users = len(set(msg['user'] for msg in messages))
        avg_message_length = np.mean([len(msg['message'].split()) for msg in messages]) if messages else 0
        
        # Временное распределение сообщений
        reaction_times = [msg['relative_time'] for msg in messages_after if msg['relative_time'] >= 0]
        avg_reaction_time = np.mean(reaction_times) if reaction_times else 0
        
        return {
            'tokenized_messages': tokenized_messages,
            'messages_count': len(messages),
            'unique_users': unique_users,
            'avg_message_length': avg_message_length,
            'avg_reaction_time': avg_reaction_time,
            'immediate_reactions': len([t for t in reaction_times if t <= 5]),
            'delayed_reactions': len([t for t in reaction_times if t > 5])
        }
    
    def calculate_reaction_intensity(self, item: Dict) -> float:
        """Вычисляет интенсивность реакции чата"""
        messages = item.get('messages', [])
        messages_after = item.get('messages_after', [])
        
        if not messages:
            return 0.0
        
        # Базовая интенсивность на основе количества сообщений
        base_intensity = len(messages_after) / 10.0  # Нормализация
        
        # Бонус за быстрые реакции
        quick_reactions = len([msg for msg in messages_after if 0 <= msg['relative_time'] <= 5])
        quick_bonus = quick_reactions / 5.0
        
        # Бонус за разнообразие пользователей
        unique_users = len(set(msg['user'] for msg in messages_after))
        diversity_bonus = unique_users / 5.0
        
        # Общая интенсивность
        intensity = min(1.0, base_intensity + quick_bonus + diversity_bonus)
        
        return intensity
    
    def classify_reaction_type(self, item: Dict) -> str:
        """Классифицирует тип реакции"""
        messages_after = item.get('messages_after', [])
        
        if not messages_after:
            return 'no_reaction'
        
        # Анализируем слова в реакциях
        all_text = ' '.join([msg['message'].lower() for msg in messages_after])
        
        # Простая классификация по ключевым словам
        positive_words = ['lol', 'haha', 'wow', 'amazing', 'great', 'poggers', 'pog', 'kappa']
        negative_words = ['wtf', 'bad', 'terrible', 'boring', 'residentsleeper']
        question_words = ['?', 'what', 'how', 'why', 'when', 'where']
        
        if any(word in all_text for word in positive_words):
            return 'positive'
        elif any(word in all_text for word in negative_words):
            return 'negative'
        elif any(word in all_text for word in question_words):
            return 'questioning'
        else:
            return 'neutral'
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        return {
            'visual_features': torch.FloatTensor(item['visual_features']),
            'tokenized_messages': torch.LongTensor(item['chat_features']['tokenized_messages']),
            'chat_stats': torch.FloatTensor([
                item['chat_features']['messages_count'],
                item['chat_features']['unique_users'],
                item['chat_features']['avg_message_length'],
                item['chat_features']['avg_reaction_time'],
                item['chat_features']['immediate_reactions'],
                item['chat_features']['delayed_reactions']
            ]),
            'reaction_intensity': torch.FloatTensor([item['reaction_intensity']]),
            'reaction_type': item['reaction_type']
        }

class MultiModalReactionModel(nn.Module):
    """Мультимодальная модель для предсказания реакций чата"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, visual_dim=20):
        super(MultiModalReactionModel, self).__init__()
        
        # Текстовая часть (LSTM для обработки чата)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.text_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Визуальная часть
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Статистики чата
        self.chat_stats_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Объединяющий слой
        combined_dim = hidden_dim * 2 + 32 + 16  # LSTM выходы + визуальные + статистики
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Выходные слои
        self.intensity_head = nn.Linear(64, 1)  # Регрессия интенсивности
        self.type_head = nn.Linear(64, 4)       # Классификация типа (positive, negative, questioning, neutral)
        
    def forward(self, visual_features, tokenized_messages, chat_stats):
        batch_size = visual_features.size(0)
        
        # Обработка текста
        embedded = self.embedding(tokenized_messages)
        text_output, (hidden, _) = self.text_lstm(embedded)
        # Используем последнее состояние из двунаправленного LSTM
        text_features = torch.cat([hidden[0], hidden[1]], dim=1)  # Concatenate forward and backward
        
        # Обработка визуальных признаков
        visual_encoded = self.visual_encoder(visual_features)
        
        # Обработка статистик чата
        chat_stats_encoded = self.chat_stats_encoder(chat_stats)
        
        # Объединение всех модальностей
        combined = torch.cat([text_features, visual_encoded, chat_stats_encoded], dim=1)
        fused = self.fusion_layer(combined)
        
        # Предсказания
        intensity = torch.sigmoid(self.intensity_head(fused))  # Интенсивность от 0 до 1
        reaction_type = self.type_head(fused)  # Логиты для классификации
        
        return intensity, reaction_type

class ReactionPredictor:
    \"\"\"Тренер и предсказатель реакций чата\"\"\"\n    \n    def __init__(self, model_save_path='reaction_model.pth'):\n        self.model_save_path = model_save_path\n        self.model = None\n        self.tokenizer = None\n        self.reaction_type_encoder = LabelEncoder()\n        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n        \n        self.setup_logging()\n    \n    def setup_logging(self):\n        logging.basicConfig(level=logging.INFO)\n        self.logger = logging.getLogger(__name__)\n    \n    def load_dataset(self, dataset_file: str) -> ChatReactionDataset:\n        \"\"\"Загружает и подготавливает датасет\"\"\"\n        self.logger.info(f\"🔍 Загружаю датасет из {dataset_file}\")\n        \n        with open(dataset_file, 'r', encoding='utf-8') as f:\n            data = json.load(f)\n        \n        training_data = data.get('training_data', [])\n        \n        # Фильтруем данные с реакциями\n        filtered_data = [item for item in training_data if item.get('messages')]\n        \n        self.logger.info(f\"✅ Загружено {len(filtered_data)} событий с реакциями\")\n        \n        dataset = ChatReactionDataset(filtered_data)\n        self.tokenizer = dataset.tokenizer\n        \n        return dataset\n    \n    def prepare_data_loaders(self, dataset: ChatReactionDataset, batch_size=32, test_size=0.2):\n        \"\"\"Подготавливает загрузчики данных для обучения и валидации\"\"\"\n        # Разделяем на обучающую и валидационную выборки\n        train_indices, val_indices = train_test_split(\n            range(len(dataset)), test_size=test_size, random_state=42\n        )\n        \n        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)\n        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)\n        \n        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)\n        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)\n        \n        return train_loader, val_loader\n    \n    def train_model(self, dataset: ChatReactionDataset, epochs=50, batch_size=32, learning_rate=0.001):\n        \"\"\"Обучает модель\"\"\"\n        self.logger.info(f\"🚀 Начинаю обучение модели на {len(dataset)} примерах\")\n        \n        # Подготавливаем энкодер типов реакций\n        all_reaction_types = [item['reaction_type'] for item in dataset.processed_data]\n        self.reaction_type_encoder.fit(all_reaction_types)\n        \n        # Создаем модель\n        vocab_size = len(self.tokenizer)\n        self.model = MultiModalReactionModel(vocab_size)\n        self.model.to(self.device)\n        \n        # Подготавливаем данные\n        train_loader, val_loader = self.prepare_data_loaders(dataset, batch_size)\n        \n        # Оптимизатор и функции потерь\n        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)\n        intensity_criterion = nn.MSELoss()\n        type_criterion = nn.CrossEntropyLoss()\n        \n        best_val_loss = float('inf')\n        \n        for epoch in range(epochs):\n            # Обучение\n            self.model.train()\n            train_loss = 0.0\n            \n            for batch in train_loader:\n                optimizer.zero_grad()\n                \n                visual_features = batch['visual_features'].to(self.device)\n                tokenized_messages = batch['tokenized_messages'].to(self.device)\n                chat_stats = batch['chat_stats'].to(self.device)\n                \n                target_intensity = batch['reaction_intensity'].to(self.device)\n                target_types = [self.reaction_type_encoder.transform([rt])[0] for rt in batch['reaction_type']]\n                target_types = torch.LongTensor(target_types).to(self.device)\n                \n                # Прямой проход\n                pred_intensity, pred_types = self.model(visual_features, tokenized_messages, chat_stats)\n                \n                # Вычисляем потери\n                intensity_loss = intensity_criterion(pred_intensity, target_intensity)\n                type_loss = type_criterion(pred_types, target_types)\n                \n                total_loss = intensity_loss + type_loss\n                \n                # Обратный проход\n                total_loss.backward()\n                optimizer.step()\n                \n                train_loss += total_loss.item()\n            \n            # Валидация\n            self.model.eval()\n            val_loss = 0.0\n            \n            with torch.no_grad():\n                for batch in val_loader:\n                    visual_features = batch['visual_features'].to(self.device)\n                    tokenized_messages = batch['tokenized_messages'].to(self.device)\n                    chat_stats = batch['chat_stats'].to(self.device)\n                    \n                    target_intensity = batch['reaction_intensity'].to(self.device)\n                    target_types = [self.reaction_type_encoder.transform([rt])[0] for rt in batch['reaction_type']]\n                    target_types = torch.LongTensor(target_types).to(self.device)\n                    \n                    pred_intensity, pred_types = self.model(visual_features, tokenized_messages, chat_stats)\n                    \n                    intensity_loss = intensity_criterion(pred_intensity, target_intensity)\n                    type_loss = type_criterion(pred_types, target_types)\n                    \n                    total_loss = intensity_loss + type_loss\n                    val_loss += total_loss.item()\n            \n            avg_train_loss = train_loss / len(train_loader)\n            avg_val_loss = val_loss / len(val_loader)\n            \n            self.logger.info(f\"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\")\n            \n            # Сохраняем лучшую модель\n            if avg_val_loss < best_val_loss:\n                best_val_loss = avg_val_loss\n                self.save_model()\n                self.logger.info(f\"💾 Сохранена лучшая модель (val_loss: {avg_val_loss:.4f})\")\n    \n    def save_model(self):\n        \"\"\"Сохраняет модель и связанные данные\"\"\"\n        save_data = {\n            'model_state_dict': self.model.state_dict(),\n            'tokenizer': self.tokenizer,\n            'reaction_type_encoder': self.reaction_type_encoder\n        }\n        \n        torch.save(save_data, self.model_save_path)\n        \n    def load_model(self, vocab_size):\n        \"\"\"Загружает сохраненную модель\"\"\"\n        if not Path(self.model_save_path).exists():\n            raise FileNotFoundError(f\"Модель не найдена: {self.model_save_path}\")\n        \n        save_data = torch.load(self.model_save_path, map_location=self.device)\n        \n        self.model = MultiModalReactionModel(vocab_size)\n        self.model.load_state_dict(save_data['model_state_dict'])\n        self.model.to(self.device)\n        \n        self.tokenizer = save_data['tokenizer']\n        self.reaction_type_encoder = save_data['reaction_type_encoder']\n        \n    def predict_reaction(self, visual_features: np.ndarray, chat_history: List[str]) -> Dict:\n        \"\"\"Предсказывает реакцию чата на событие\"\"\"\n        if self.model is None:\n            raise ValueError(\"Модель не загружена. Сначала обучите или загрузите модель.\")\n        \n        self.model.eval()\n        \n        # Подготавливаем данные\n        visual_tensor = torch.FloatTensor(visual_features).unsqueeze(0).to(self.device)\n        \n        # Токенизируем историю чата\n        chat_text = ' '.join(chat_history)\n        tokenized = self.tokenize_text_simple(chat_text)\n        chat_tensor = torch.LongTensor(tokenized).unsqueeze(0).to(self.device)\n        \n        # Простые статистики чата\n        chat_stats = torch.FloatTensor([\n            len(chat_history),  # количество сообщений\n            len(set(chat_history)),  # уникальные \"пользователи\" (упрощение)\n            np.mean([len(msg.split()) for msg in chat_history]) if chat_history else 0,\n            0, 0, 0  # остальные статистики заполняем нулями\n        ]).unsqueeze(0).to(self.device)\n        \n        with torch.no_grad():\n            pred_intensity, pred_types = self.model(visual_tensor, chat_tensor, chat_stats)\n            \n            intensity = pred_intensity.cpu().numpy()[0][0]\n            type_probs = F.softmax(pred_types, dim=1).cpu().numpy()[0]\n            predicted_type = self.reaction_type_encoder.inverse_transform([np.argmax(type_probs)])[0]\n        \n        return {\n            'predicted_intensity': float(intensity),\n            'predicted_type': predicted_type,\n            'type_probabilities': {\n                label: float(prob) for label, prob in \n                zip(self.reaction_type_encoder.classes_, type_probs)\n            }\n        }\n    \n    def tokenize_text_simple(self, text: str, max_length=512) -> List[int]:\n        \"\"\"Простая токенизация текста\"\"\"\n        words = text.lower().split()\n        tokens = [self.tokenizer.get(word, 1) for word in words]  # 1 = <UNK>\n        \n        tokens = [2] + tokens + [3]  # START and END tokens\n        \n        if len(tokens) > max_length:\n            tokens = tokens[:max_length]\n        else:\n            tokens.extend([0] * (max_length - len(tokens)))\n        \n        return tokens

def main():\n    \"\"\"Главная функция для обучения модели\"\"\"\n    import argparse\n    \n    parser = argparse.ArgumentParser(description='Обучение нейронной модели предсказания реакций чата')\n    parser.add_argument('--dataset', required=True, help='Файл с датасетом')\n    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох обучения')\n    parser.add_argument('--batch-size', type=int, default=32, help='Размер батча')\n    parser.add_argument('--learning-rate', type=float, default=0.001, help='Скорость обучения')\n    parser.add_argument('--model-path', default='reaction_model.pth', help='Путь для сохранения модели')\n    \n    args = parser.parse_args()\n    \n    print(\"🧠 Тренировка нейронной модели предсказания реакций чата\")\n    print(\"=\" * 60)\n    \n    try:\n        # Создаем предсказатель\n        predictor = ReactionPredictor(args.model_path)\n        \n        # Загружаем датасет\n        dataset = predictor.load_dataset(args.dataset)\n        \n        # Обучаем модель\n        predictor.train_model(\n            dataset, \n            epochs=args.epochs, \n            batch_size=args.batch_size, \n            learning_rate=args.learning_rate\n        )\n        \n        print(f\"✅ Модель обучена и сохранена: {args.model_path}\")\n        \n    except Exception as e:\n        print(f\"❌ Ошибка: {e}\")\n        import traceback\n        traceback.print_exc()\n\nif __name__ == \"__main__\":\n    main()
