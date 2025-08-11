#!/usr/bin/env python3
"""
Запуск транскрибации аудио - отдельный модуль для PyCharm
Этот файл предназначен для независимого запуска системы транскрибации
"""
import sys
import os
from pathlib import Path
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Добавляем путь к модулям транскрибации
sys.path.append(str(Path(__file__).parent / "transcription_tools"))

try:
    from transcribe_stream import StreamTranscriber
except ImportError:
    print("ОШИБКА: Не удается импортировать модуль транскрибации")
    print("Убедитесь, что установлены зависимости:")
    print("pip install openai-whisper torch torchaudio")
    sys.exit(1)


class TranscriptionGUI:
    """Простой GUI для транскрибации"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("VTuber AI - Транскрибация аудио")
        self.root.geometry("600x400")
        
        self.audio_file = tk.StringVar()
        self.model_size = tk.StringVar(value="medium")
        self.language = tk.StringVar(value="auto")
        self.output_path = tk.StringVar()
        
        self.create_widgets()
        
    def create_widgets(self):
        # Заголовок
        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=10)
        
        ttk.Label(title_frame, text="🎙️ Транскрибация аудио стримов", 
                 font=("Arial", 16, "bold")).pack()
        
        # Выбор файла
        file_frame = ttk.LabelFrame(self.root, text="Выбор аудиофайла", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Entry(file_frame, textvariable=self.audio_file, width=50).pack(side="left", fill="x", expand=True)
        ttk.Button(file_frame, text="Обзор...", command=self.select_file).pack(side="right", padx=(5, 0))
        
        # Настройки
        settings_frame = ttk.LabelFrame(self.root, text="Настройки транскрибации", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Размер модели
        ttk.Label(settings_frame, text="Размер модели:").grid(row=0, column=0, sticky="w", pady=2)
        model_combo = ttk.Combobox(settings_frame, textvariable=self.model_size, 
                                  values=["tiny", "base", "small", "medium", "large"],
                                  width=15)
        model_combo.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=2)
        
        # Язык
        ttk.Label(settings_frame, text="Язык:").grid(row=1, column=0, sticky="w", pady=2)
        lang_combo = ttk.Combobox(settings_frame, textvariable=self.language,
                                 values=["auto", "ru", "en", "de", "fr", "es", "ja", "zh"],
                                 width=15)
        lang_combo.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=2)
        
        # Путь сохранения
        output_frame = ttk.LabelFrame(self.root, text="Сохранение результата", padding=10)
        output_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side="left", fill="x", expand=True)
        ttk.Button(output_frame, text="Выбрать...", command=self.select_output).pack(side="right", padx=(5, 0))
        
        # Кнопки управления
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=20)
        
        ttk.Button(control_frame, text="🚀 Начать транскрибацию", 
                  command=self.start_transcription, style="Accent.TButton").pack(side="left", padx=5)
        ttk.Button(control_frame, text="❌ Закрыть", 
                  command=self.root.quit).pack(side="left", padx=5)
        
        # Лог
        log_frame = ttk.LabelFrame(self.root, text="Процесс транскрибации", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Прогресс бар
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill="x", padx=10, pady=(0, 10))
        
    def select_file(self):
        filename = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[
                ("Аудио файлы", "*.mp3 *.wav *.flac *.m4a *.ogg *.wma"),
                ("Все файлы", "*.*")
            ]
        )
        if filename:
            self.audio_file.set(filename)
            
    def select_output(self):
        filename = filedialog.asksaveasfilename(
            title="Сохранить результат как",
            defaultextension=".json",
            filetypes=[
                ("JSON файлы", "*.json"),
                ("Все файлы", "*.*")
            ]
        )
        if filename:
            self.output_path.set(filename)
            
    def log(self, message):
        """Добавить сообщение в лог"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def start_transcription(self):
        """Запуск процесса транскрибации"""
        if not self.audio_file.get():
            messagebox.showerror("Ошибка", "Выберите аудиофайл!")
            return
            
        if not os.path.exists(self.audio_file.get()):
            messagebox.showerror("Ошибка", "Выбранный файл не существует!")
            return
            
        try:
            self.progress.start()
            self.log("🚀 Начинаю транскрибацию...")
            
            # Определяем параметры
            model_size = self.model_size.get()
            language = self.language.get() if self.language.get() != "auto" else None
            audio_path = self.audio_file.get()
            
            # Определяем путь сохранения
            if self.output_path.get():
                output_path = self.output_path.get()
            else:
                filename = Path(audio_path).stem + '_transcript.json'
                output_dir = Path(__file__).parent / "transcription_tools" / "data"
                output_dir.mkdir(exist_ok=True)
                output_path = str(output_dir / filename)
            
            self.log(f"📄 Файл: {Path(audio_path).name}")
            self.log(f"🧠 Модель: {model_size}")
            self.log(f"🌍 Язык: {language or 'автоопределение'}")
            self.log(f"💾 Результат: {output_path}")
            
            # Создаем транскрибер
            self.log("⬇️ Загрузка модели Whisper...")
            transcriber = StreamTranscriber(model_size=model_size)
            
            # Выполняем транскрибацию
            self.log("🎯 Выполняем транскрибацию...")
            transcription = transcriber.transcribe_audio(audio_path, language=language)
            
            # Сохраняем результат
            transcriber.save_transcription(transcription, output_path)
            transcriber.save_text_only(transcription, output_path)
            
            # Выводим статистику
            self.log("✅ Транскрибация завершена!")
            self.log(f"📊 Язык: {transcription.get('language', 'не определен')}")
            self.log(f"⏱️ Длительность: {transcription.get('duration', 0):.2f} сек")
            self.log(f"📝 Сегментов: {len(transcription.get('segments', []))}")
            
            # Показываем превью
            text = transcription.get('text', '')
            preview = text[:100] + '...' if len(text) > 100 else text
            self.log(f"👁️ Превью: {preview}")
            
            messagebox.showinfo("Готово!", f"Транскрибация завершена!\nРезультат сохранен в:\n{output_path}")
            
        except Exception as e:
            self.log(f"❌ Ошибка: {str(e)}")
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")
        finally:
            self.progress.stop()


def run_cli():
    """Запуск в режиме командной строки"""
    parser = argparse.ArgumentParser(description='Транскрибация аудио - CLI режим')
    parser.add_argument('audio_path', nargs='?', help='Путь к аудиофайлу')
    parser.add_argument('--model', '-m', default='medium', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Размер модели Whisper')
    parser.add_argument('--language', '-l', help='Код языка (ru, en, etc.)')
    parser.add_argument('--output', '-o', help='Путь для сохранения результата')
    parser.add_argument('--gui', action='store_true', help='Запустить графический интерфейс')
    
    args = parser.parse_args()
    
    # Если указан --gui или нет аргументов, запускаем GUI
    if args.gui or not args.audio_path:
        print("🖥️ Запуск графического интерфейса...")
        root = tk.Tk()
        app = TranscriptionGUI(root)
        root.mainloop()
        return
    
    # CLI режим
    try:
        transcriber = StreamTranscriber(model_size=args.model)
        transcription = transcriber.transcribe_audio(args.audio_path, language=args.language)
        
        # Определяем путь сохранения
        if args.output:
            output_path = args.output
        else:
            filename = Path(args.audio_path).stem + '_transcript.json'
            output_dir = Path(__file__).parent / "transcription_tools" / "data"
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / filename)
        
        transcriber.save_transcription(transcription, output_path)
        transcriber.save_text_only(transcription, output_path)
        transcriber.print_summary(transcription)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


def main():
    """Главная функция"""
    print("🎙️ VTuber AI - Система транскрибации")
    print("=" * 50)
    
    # Проверяем зависимости
    try:
        import whisper
        import torch
        print("✅ Все зависимости установлены")
    except ImportError as e:
        print(f"❌ Отсутствует зависимость: {e}")
        print("\n🔧 Для установки выполните:")
        print("pip install openai-whisper torch torchaudio")
        return
    
    # Проверяем доступность GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU доступен: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ GPU недоступен, будет использоваться CPU (медленно)")
    
    run_cli()


if __name__ == "__main__":
    main()
