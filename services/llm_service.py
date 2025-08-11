"""
LLM микросервис для обработки текстовых запросов
Отдельный процесс для работы с языковой моделью
"""
import asyncio
import logging
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
import os
from pathlib import Path

# Добавляем родительскую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))
from config.microservices import SERVICES

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Service", description="Сервис обработки текста")

class ChatRequest(BaseModel):
    message: str
    context: str = ""
    max_length: int = 100
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    emotion: str = "neutral"
    confidence: float = 1.0

class LLMHandler:
    """Обработчик языковой модели"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
        logger.info(f"LLM Handler initialized with device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
    async def load_model(self, model_name: str = "sberbank-ai/rugpt3large_based_on_gpt2"):
        """Загружает модель"""
        try:
            logger.info(f"Загрузка модели {model_name} на {self.device}")
            
            # Пробуем загрузить основную модель, если не получается - fallback на меньшую
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                # Оптимизируем для GPU
                if self.device == "cuda":
                    self.model = GPT2LMHeadModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,  # Используем half precision для экономии памяти
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                else:
                    self.model = GPT2LMHeadModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32
                    )
            except Exception as e:
                logger.warning(f"Не удалось загрузить {model_name}: {e}")
                logger.info("Загружаю меньшую модель gpt2...")
                model_name = "gpt2"
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                self.model = GPT2LMHeadModel.from_pretrained(model_name)
            
            # Устанавливаем pad_token, если отсутствует
            if self.tokenizer.pad_token is None:
                # Используем разный токен для pad, чтобы избежать конфликта с eos
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # Добавляем специальный токен для лучшей работы
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
                
            self.model.to(self.device)
            self.loaded = True
            
            logger.info(f"Модель {model_name} успешно загружена на {self.device}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    async def generate_response(self, message: str, context: str = "", 
                              max_length: int = 100, temperature: float = 0.7) -> dict:
        """Генерирует ответ на сообщение"""
        if not self.loaded:
            raise HTTPException(status_code=503, detail="Модель не загружена")
            
        try:
            # Промпт в стиле Neuro-sama
            vtuber_context = f"""Нейро-чан - это умная и остроумная AI VTuber. Она очень любит игры, аниме и общение с чатом. Она часто сарказмична, но в то же время добрая и весёлая. Она отвечает как настоящий человек, с эмоциями и своим мнением.

Пользователь: {message}
Нейро-чан:"""
            
            prompt = vtuber_context
            logger.info(f"Generated prompt: {prompt}")
            
            # Токенизация с attention_mask
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Генерация с улучшенными параметрами
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=inputs["input_ids"].shape[1] + max_length,
                    temperature=0.3,  # Низкая температура для меньших галлюцинаций
                    do_sample=True,
                    top_p=0.7,  # Более строгий отбор
                    top_k=40,  # Ограничиваем выбор слов
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Увеличиваем штраф за повторы
                    no_repeat_ngram_size=2,
                    early_stopping=True,  # Ранняя остановка
                    eos_token_id=self.tokenizer.eos_token_id,
                    bad_words_ids=[[self.tokenizer.encode("П")[0]]]  # Предотвращаем повторы с "Пользователь"
                )
            
            # Декодирование
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Full generated response: {full_response}")
            
            # Извлекаем только новый сгенерированный текст
            original_length = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][original_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            logger.info(f"Extracted response: {response}")
            
            # Если ответ пустой, пробуем альтернативную логику
            if not response:
                # Извлекаем только ответ Нейро-чан из полного ответа
                if "Нейро-чан:" in full_response:
                    parts = full_response.split("Нейро-чан:")
                    if len(parts) > 1:
                        response = parts[-1].strip()
                        logger.info(f"Alternative extraction: {response}")
            
            # Очищаем ответ от лишних символов и галлюцинаций
            if response:
                # Убираем переносы строк
                response = response.replace('\n', ' ').strip()
                
                # Останавливаемся на первом предложении
                sentences = response.split('.')
                if len(sentences) > 1 and sentences[0].strip():
                    response = sentences[0].strip() + '.'
                
                # Останавливаемся, если встречаем начало нового диалога
                stop_words = ['Пользователь:', 'П:', 'Н:', 'Нейро-чан:', 'Блоггер:', 'Парень:']
                for stop_word in stop_words:
                    if stop_word in response:
                        response = response.split(stop_word)[0].strip()
                
                # Обрезаем слишком длинные ответы
                if len(response) > 100:
                    response = response[:100].strip()
                    if not response.endswith(('.', '!', '?')):
                        response += '...'
                
                # Убираем HTML символы
                response = response.replace('&quot;', '"').replace('&laquo;', '«').replace('&raquo;', '»').replace('&nbsp;', ' ')
                
                # Очищаем от служебных символов
                response = response.strip('"«»')
            
            # Если все еще пустой, даем простой ответ
            if not response or len(response) < 3:
                response = "Привет! Очень приятно познакомиться! (◕‿‿◕)"
                logger.warning("Generated empty or too short response, using fallback")
            
            # Определяем эмоцию (простейший вариант)
            emotion = self._detect_emotion(response)
            
            return {
                "response": response,
                "emotion": emotion,
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _detect_emotion(self, text: str) -> str:
        """Определение эмоции по русским ключевым словам"""
        text_lower = text.lower()
        
        # Восхищение/Возбуждение
        if any(word in text_lower for word in ["!", "вау", "классно", "здорово", "круто", "офигенно"]):
            return "excited"
        # Удивление/Замешательство
        elif any(word in text_lower for word in ["?", "хм", "что", "как", "почему", "не понял"]):
            return "confused"
        # Счастье/Смех
        elif any(word in text_lower for word in ["хаха", "лол", "кек", "смешно", "хехе", "ня~", "я рада"]):
            return "happy"
        # Грусть/Извинения
        elif any(word in text_lower for word in ["извини", "грустно", "плохо", "ошибка", "прости"]):
            return "sad"
        # Мышление
        elif any(word in text_lower for word in ["думаю", "размышляю", "может быть", "наверное"]):
            return "thinking"
        else:
            return "neutral"

# Глобальный экземпляр обработчика
llm_handler = LLMHandler()

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    await llm_handler.load_model()

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "model_loaded": llm_handler.loaded,
        "device": llm_handler.device
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Обработка чат-запроса"""
    result = await llm_handler.generate_response(
        message=request.message,
        context=request.context,
        max_length=request.max_length,
        temperature=request.temperature
    )
    
    return ChatResponse(**result)

@app.get("/status")
async def get_status():
    """Статус модели"""
    if not llm_handler.loaded:
        return {"status": "model_not_loaded"}
    
    status = {
        "status": "ready",
        "device": llm_handler.device,
    }
    
    if torch.cuda.is_available():
        status.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_cached_gb": torch.cuda.memory_reserved() / 1024**3,
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "memory_usage_percent": (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
        })
    else:
        status.update({
            "memory_allocated": 0,
            "memory_cached": 0
        })
    
    return status

if __name__ == "__main__":
    service_config = SERVICES["llm_service"]
    uvicorn.run(
        app,
        host=service_config["host"],
        port=service_config["port"],
        log_level="info"
    )
