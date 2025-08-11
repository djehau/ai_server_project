#!/usr/bin/env python3
"""
Продвинутый анализатор событий на экране
Детектирует различные типы событий: лица, эмоции, текст, движение
"""
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

class AdvancedEventDetector:
    """Продвинутый детектор событий на экране"""
    
    def __init__(self):
        self.setup_logging()
        self.load_cascades()
        
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_cascades(self):
        """Загружает каскады Хаара для детекции лиц"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.logger.info("✅ Каскады для детекции загружены")
        except Exception as e:
            self.logger.warning(f"⚠️  Не удалось загрузить каскады: {e}")
            self.face_cascade = None
    
    def detect_faces_and_emotions(self, frame_path: str) -> List[Dict]:
        """
        Детектирует лица и базовые эмоции на кадре
        
        Args:
            frame_path: Путь к кадру
            
        Returns:
            Список найденных лиц с их характеристиками
        """
        if not self.face_cascade:
            return []
            
        img = cv2.imread(frame_path)
        if img is None:
            return []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        detected_faces = []
        for (x, y, w, h) in faces:
            face_roi_gray = gray[y:y+h, x:x+w]
            
            # Детектируем улыбки
            smiles = self.smile_cascade.detectMultiScale(face_roi_gray, 1.8, 20)
            has_smile = len(smiles) > 0
            
            # Детектируем глаза
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
            eyes_count = len(eyes)
            
            detected_faces.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'has_smile': has_smile,
                'smile_confidence': len(smiles) / 10.0 if smiles is not None else 0.0,
                'eyes_detected': eyes_count,
                'face_area': int(w * h),
                'center_x': int(x + w/2),
                'center_y': int(y + h/2)
            })
        
        return detected_faces
    
    def detect_text_regions(self, frame_path: str) -> List[Dict]:
        """
        Детектирует текстовые области на кадре
        
        Args:
            frame_path: Путь к кадру
            
        Returns:
            Список найденных текстовых областей
        """
        img = cv2.imread(frame_path)
        if img is None:
            return []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Используем EAST text detector (если доступен) или простой метод
        # Простая детекция через контуры и морфологические операции
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Бинаризация
        _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Соединяем близкие элементы
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        
        # Находим контуры
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Фильтруем по размеру (возможный текст)
            if w > 20 and h > 10 and w > h:
                aspect_ratio = w / h
                area = w * h
                
                text_regions.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'area': int(area),
                    'aspect_ratio': float(aspect_ratio),
                    'center_x': int(x + w/2),
                    'center_y': int(y + h/2)
                })
        
        return text_regions
    
    def detect_color_changes(self, frame_path: str, prev_frame_path: str = None) -> Dict:
        """
        Детектирует изменения в цветовой палитре кадра
        
        Args:
            frame_path: Текущий кадр
            prev_frame_path: Предыдущий кадр для сравнения
            
        Returns:
            Информация об изменениях цветов
        """
        img = cv2.imread(frame_path)
        if img is None:
            return {}
            
        # Конвертируем в HSV для лучшего анализа цветов
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Анализируем гистограммы цветов
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Находим доминирующие цвета
        dominant_hue = np.argmax(hist_h)
        dominant_saturation = np.argmax(hist_s)
        dominant_value = np.argmax(hist_v)
        
        # Вычисляем среднюю яркость
        avg_brightness = np.mean(hsv[:, :, 2])
        
        color_info = {
            'dominant_hue': int(dominant_hue),
            'dominant_saturation': int(dominant_saturation),
            'dominant_value': int(dominant_value),
            'avg_brightness': float(avg_brightness),
            'color_diversity': float(np.std(hist_h))
        }
        
        # Если есть предыдущий кадр, сравниваем
        if prev_frame_path:
            prev_img = cv2.imread(prev_frame_path)
            if prev_img is not None:
                prev_hsv = cv2.cvtColor(prev_img, cv2.COLOR_BGR2HSV)
                prev_brightness = np.mean(prev_hsv[:, :, 2])
                
                color_info['brightness_change'] = float(avg_brightness - prev_brightness)
                
                # Сравниваем гистограммы
                prev_hist_h = cv2.calcHist([prev_hsv], [0], None, [180], [0, 180])
                correlation = cv2.compareHist(hist_h, prev_hist_h, cv2.HISTCMP_CORREL)
                color_info['color_similarity'] = float(correlation)
        
        return color_info
    
    def detect_motion_intensity(self, frame_path: str, prev_frame_path: str = None) -> Dict:
        """
        Детектирует интенсивность движения между кадрами
        
        Args:
            frame_path: Текущий кадр
            prev_frame_path: Предыдущий кадр
            
        Returns:
            Информация о движении
        """
        if not prev_frame_path:
            return {'motion_intensity': 0.0}
            
        current = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        previous = cv2.imread(prev_frame_path, cv2.IMREAD_GRAYSCALE)
        
        if current is None or previous is None:
            return {'motion_intensity': 0.0}
        
        # Вычисляем оптический поток
        flow = cv2.calcOpticalFlowPyrLK(
            previous, current, 
            np.array([]), None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Простая разность кадров для общего движения
        diff = cv2.absdiff(current, previous)
        motion_intensity = np.mean(diff) / 255.0
        
        # Детектируем области с максимальным движением
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        motion_pixels = np.sum(thresh == 255)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_ratio = motion_pixels / total_pixels
        
        return {
            'motion_intensity': float(motion_intensity),
            'motion_ratio': float(motion_ratio),
            'active_pixels': int(motion_pixels)
        }
    
    def classify_event_type(self, faces: List[Dict], text_regions: List[Dict], 
                          color_info: Dict, motion_info: Dict) -> List[str]:
        """
        Классифицирует тип события на основе всех детекторов
        
        Args:
            faces: Найденные лица
            text_regions: Текстовые области
            color_info: Информация о цветах
            motion_info: Информация о движении
            
        Returns:
            Список типов событий
        """
        event_types = []
        
        # Событие с лицом
        if faces:
            event_types.append('face_detected')
            
            # Эмоциональное событие
            for face in faces:
                if face.get('has_smile', False):
                    event_types.append('positive_emotion')
                    break
        
        # Текстовое событие
        if len(text_regions) > 3:
            event_types.append('text_heavy')
        elif text_regions:
            event_types.append('text_present')
        
        # Цветовые события
        if color_info.get('brightness_change', 0) > 30:
            event_types.append('brightness_change')
        
        if color_info.get('color_similarity', 1.0) < 0.7:
            event_types.append('color_shift')
        
        # События движения
        if motion_info.get('motion_intensity', 0) > 0.3:
            event_types.append('high_motion')
        elif motion_info.get('motion_intensity', 0) > 0.1:
            event_types.append('medium_motion')
        
        # Комбинированные события
        if 'face_detected' in event_types and 'high_motion' in event_types:
            event_types.append('dynamic_face')
        
        if not event_types:
            event_types.append('static_scene')
        
        return event_types
    
    def analyze_frame_comprehensive(self, frame_path: str, prev_frame_path: str = None) -> Dict:
        """
        Комплексный анализ кадра
        
        Args:
            frame_path: Путь к кадру
            prev_frame_path: Путь к предыдущему кадру
            
        Returns:
            Полная информация о событиях на кадре
        """
        # Выполняем все виды детекции
        faces = self.detect_faces_and_emotions(frame_path)
        text_regions = self.detect_text_regions(frame_path)
        color_info = self.detect_color_changes(frame_path, prev_frame_path)
        motion_info = self.detect_motion_intensity(frame_path, prev_frame_path)
        
        # Классифицируем событие
        event_types = self.classify_event_type(faces, text_regions, color_info, motion_info)
        
        # Вычисляем общую "интересность" кадра
        interest_score = 0.0
        interest_score += len(faces) * 0.3  # Лица важны
        interest_score += len(text_regions) * 0.1  # Текст менее важен
        interest_score += motion_info.get('motion_intensity', 0) * 0.4  # Движение важно
        interest_score += (1.0 - color_info.get('color_similarity', 1.0)) * 0.2  # Изменения цвета
        
        return {
            'frame_path': frame_path,
            'faces': faces,
            'text_regions': text_regions,
            'color_info': color_info,
            'motion_info': motion_info,
            'event_types': event_types,
            'interest_score': float(interest_score),
            'total_faces': len(faces),
            'total_text_regions': len(text_regions),
            'has_faces': len(faces) > 0,
            'has_text': len(text_regions) > 0,
            'has_motion': motion_info.get('motion_intensity', 0) > 0.1
        }

def enhance_dataset_with_events(dataset_file: str, frames_dir: str, output_file: str):
    """
    Улучшает существующий датасет продвинутым анализом событий
    
    Args:
        dataset_file: Файл с базовым датасетом
        frames_dir: Папка с кадрами
        output_file: Выходной файл с улучшенным датасетом
    """
    detector = AdvancedEventDetector()
    
    print("🔍 Загружаю базовый датасет...")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    training_data = dataset.get('training_data', [])
    enhanced_data = []
    
    print(f"📊 Анализирую {len(training_data)} событий...")
    
    for i, event_data in enumerate(training_data):
        event = event_data['event']
        frame_path = event.get('frame_path')
        
        if frame_path and Path(frame_path).exists():
            # Находим предыдущий кадр для сравнения
            prev_frame_path = None
            if i > 0:
                prev_event = training_data[i-1]['event']
                prev_frame_path = prev_event.get('frame_path')
                if prev_frame_path and not Path(prev_frame_path).exists():
                    prev_frame_path = None
            
            # Выполняем продвинутый анализ
            frame_analysis = detector.analyze_frame_comprehensive(frame_path, prev_frame_path)
            
            # Объединяем с существующими данными
            enhanced_event = event_data.copy()
            enhanced_event['advanced_analysis'] = frame_analysis
            
            # Добавляем метрики для обучения
            enhanced_event['features'] = {
                'interest_score': frame_analysis['interest_score'],
                'event_complexity': len(frame_analysis['event_types']),
                'visual_elements': {
                    'faces': frame_analysis['total_faces'],
                    'text_regions': frame_analysis['total_text_regions'],
                    'motion_level': frame_analysis['motion_info'].get('motion_intensity', 0)
                }
            }
            
            enhanced_data.append(enhanced_event)
        else:
            # Оставляем без изменений, если кадр не найден
            enhanced_data.append(event_data)
        
        if (i + 1) % 50 == 0:
            print(f"⏳ Обработано {i + 1}/{len(training_data)} событий...")
    
    # Сохраняем улучшенный датасет
    enhanced_dataset = dataset.copy()
    enhanced_dataset['training_data'] = enhanced_data
    enhanced_dataset['metadata']['enhanced_analysis'] = True
    enhanced_dataset['metadata']['enhancement_version'] = '1.0'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Улучшенный датасет сохранен: {output_file}")
    
    # Статистика по типам событий
    event_type_stats = {}
    for event_data in enhanced_data:
        if 'advanced_analysis' in event_data:
            for event_type in event_data['advanced_analysis']['event_types']:
                event_type_stats[event_type] = event_type_stats.get(event_type, 0) + 1
    
    print(f"\n📈 Статистика типов событий:")
    for event_type, count in sorted(event_type_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"   {event_type}: {count}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Использование: python advanced_event_detector.py <dataset.json> <frames_dir> <output.json>")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    frames_dir = sys.argv[2] 
    output_file = sys.argv[3]
    
    enhance_dataset_with_events(dataset_file, frames_dir, output_file)
