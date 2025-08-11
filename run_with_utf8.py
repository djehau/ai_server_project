#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обертка для запуска процессов с правильной кодировкой UTF-8
"""
import os
import sys
import subprocess

# Устанавливаем кодировку UTF-8 для всех операций ввода-вывода
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '0'

# Устанавливаем кодовую страницу консоли в UTF-8
if sys.platform == 'win32':
    try:
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
    except:
        pass

# Импортируем и запускаем основной скрипт
if __name__ == "__main__":
    if len(sys.argv) > 1:
        script_name = sys.argv[1]
        
        # Импортируем и выполняем нужный модуль
        if script_name == "process_manager":
            import process_manager
        elif script_name == "main_server":
            import main_server
        else:
            print(f"Неизвестный скрипт: {script_name}")
            sys.exit(1)
    else:
        # По умолчанию запускаем process_manager
        import process_manager
