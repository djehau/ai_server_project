"""
Fine-tuning модуля для создания VTuber AI
Специализированная обучение на VTuber контенте
"""

import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import logging
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VTuberDataset:
    """Датасет для обучения VTuber модели"""
    
    def __init__(self, data_path: str, tokenizer: GPT2Tokenizer):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.conversations = []
        self.emotions = [
            "happy", "excited", "sad", "angry", "confused", 
            "thinking", "surprised", "neutral", "playful", "sarcastic"
        ]
    
    def load_vtuber_data(self):
        """Загружает данные VTuber диалогов"""
        # Пример структуры данных
        sample_data = [
            {
                "context": "Игрок играет в сложную игру и постоянно умирает",
                "user_message": "Это игра такая сложная!",
                "vtuber_response": "Ой, да я знаю! Эта игра просто издевается над нами! Но не сдавайся, ты справишься! 💪",
                "emotion": "encouraging",
                "game_context": "difficult_game"
            },
            {
                "context": "Обычный чат во время стрима",
                "user_message": "Привет Нейро-чан!",
                "vtuber_response": "Привет! Как дела? Рада тебя видеть в чате! ✨",
                "emotion": "happy",
                "game_context": "chat"
            },
            {
                "context": "Игрок спрашивает о любимых играх",
                "user_message": "Какие игры тебе нравятся?",
                "vtuber_response": "Ох, я обожаю RPG и инди-игры! Особенно те, где можно исследовать мир и встречать интересных персонажей. А тебе какие нравятся?",
                "emotion": "excited",
                "game_context": "discussion"
            }
        ]
        
        # Сохраняем пример данных
        with open(self.data_path / "vtuber_training_data.json", "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        return sample_data
    
    def create_training_prompt(self, item: Dict[str, Any]) -> str:
        """Создает промпт для обучения"""
        context = item.get("context", "")
        user_msg = item["user_message"]
        response = item["vtuber_response"]
        emotion = item.get("emotion", "neutral")
        
        prompt = f"""Контекст: {context}
Эмоция: {emotion}
Пользователь: {user_msg}
Нейро-чан: {response}<|endoftext|>"""
        
        return prompt
    
    def prepare_dataset(self) -> Dataset:
        """Подготавливает датасет для обучения"""
        data = self.load_vtuber_data()
        
        # Создаем промпты
        prompts = []
        for item in data:
            prompt = self.create_training_prompt(item)
            prompts.append(prompt)
        
        # Токенизация
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Создаем dataset
        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"]
        })
        
        return dataset

class VTuberTrainer:
    """Тренер для VTuber модели"""
    
    def __init__(self, model_name: str = "sberbank-ai/rugpt3large_based_on_gpt2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Загружает базовую модель"""
        logger.info(f"Загружаем модель: {self.model_name}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Добавляем специальные токены
        special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|endoftext|>",
            "additional_special_tokens": ["<|emotion|>", "<|context|>", "<|user|>", "<|vtuber|>"]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        
    def train(self, dataset: Dataset, output_dir: str = "vtuber_model"):
        """Обучает модель"""
        
        # Настройки обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Маленький batch для 12GB
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=True,  # Используем mixed precision
            dataloader_pin_memory=False,
            learning_rate=5e-5,
            weight_decay=0.01,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            logging_dir="./logs",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Создаем trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Обучаем
        logger.info("Начинаем обучение...")
        trainer.train()
        
        # Сохраняем модель
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Модель сохранена в {output_dir}")

def main():
    """Основная функция для запуска обучения"""
    
    # Создаем папку для обучения
    training_dir = Path("training_data")
    training_dir.mkdir(exist_ok=True)
    
    # Инициализируем trainer
    trainer = VTuberTrainer()
    trainer.load_model()
    
    # Подготавливаем данные
    dataset_creator = VTuberDataset(training_dir, trainer.tokenizer)
    dataset = dataset_creator.prepare_dataset()
    
    # Обучаем
    trainer.train(dataset)
    
    print("✅ Обучение завершено!")

if __name__ == "__main__":
    main()
