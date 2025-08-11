"""
Основной сервер VTuber AI - координатор всех микросервисов
"""
from flask import Flask, render_template, jsonify, request, redirect
from flask_socketio import SocketIO, emit
import requests
import logging
import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))
from config.microservices import SERVICES

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vtuber_ai_secret'
app.config['JSON_AS_ASCII'] = False  # Разрешаем не-ASCII символы в JSON
socketio = SocketIO(app, cors_allowed_origins="*")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceManager:
    """Менеджер для взаимодействия с микросервисами"""
    
    def __init__(self):
        self.services = SERVICES
    
    def get_service_url(self, service_name: str) -> str:
        """Получает URL сервиса"""
        if service_name in self.services:
            config = self.services[service_name]
            return f"http://{config['host']}:{config['port']}"
        return None
    
    def check_service_health(self, service_name: str) -> bool:
        """Проверяет здоровье сервиса"""
        try:
            url = self.get_service_url(service_name)
            if url:
                response = requests.get(f"{url}/health", timeout=5)
                return response.status_code == 200
        except:
            return False
        return False
    
    def get_all_services_status(self) -> dict:
        """Получает статус всех сервисов"""
        status = {}
        for service_name in self.services:
            status[service_name] = {
                "running": self.check_service_health(service_name),
                "url": self.get_service_url(service_name),
                "description": self.services[service_name]["description"]
            }
        return status

service_manager = ServiceManager()

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "main_server"})

@app.route('/status')
def status():
    """API статуса всех сервисов"""
    return jsonify(service_manager.get_all_services_status())

@app.route('/metrics')
def metrics():
    """Получение метрик системы"""
    try:
        monitor_url = service_manager.get_service_url('monitor_service')
        if monitor_url:
            response = requests.get(f"{monitor_url}/metrics", timeout=5)
            if response.status_code == 200:
                return jsonify(response.json())
        return jsonify({"error": "Monitor service unavailable"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/logs/debug')
def logs_debug():
    """Диагностика логов"""
    log_path = Path(__file__).parent / "logs" / "process_manager.log"
    current_dir = Path(__file__).parent
    logs_dir = current_dir / "logs"
    
    return jsonify({
        "current_dir": str(current_dir),
        "logs_dir": str(logs_dir),
        "logs_dir_exists": logs_dir.exists(),
        "log_path": str(log_path),
        "log_file_exists": log_path.exists(),
        "files_in_logs": list(logs_dir.glob("*")) if logs_dir.exists() else []
    })

@app.route('/logs')
def get_logs():
    """Получение логов процесса"""
    try:
        # Получаем абсолютный путь к файлу логов
        log_path = Path(__file__).parent / "logs" / "process_manager.log"
        
        if not log_path.exists():
            return jsonify({"error": f"Файл логов не найден: {log_path}"})
        
        # Читаем файл как байты и пробуем разные кодировки
        with open(log_path, "rb") as f:
            content = f.read()
        
        # Пробуем разные кодировки
        encodings = ['utf-8', 'cp1251', 'windows-1251', 'cp866', 'latin1']
        
        for encoding in encodings:
            try:
                text = content.decode(encoding)
                # Проверяем, что декодирование прошло успешно
                if 'РзРаРїС„С‰С‰РµРЅ' in text or 'Запущен' in text:
                    lines = text.splitlines(keepends=True)
                    last_lines = lines[-20:] if lines else []
                    return jsonify({"logs": last_lines})
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Если ничего не подошло, возвращаем как есть с replace
        try:
            text = content.decode('utf-8', errors='replace')
            lines = text.splitlines(keepends=True)
            last_lines = lines[-20:] if lines else []
        except:
            # Крайний случай - отображаем hex
            last_lines = [f"Binary content (length: {len(content)} bytes)"]
        
        return jsonify({"logs": last_lines})
        
    except FileNotFoundError:
        return jsonify({"error": "Файл логов не найден"})
    except PermissionError:
        return jsonify({"error": "Нет доступа к файлу логов"})
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return jsonify({"error": f"Ошибка: {str(e)}"})

@app.route('/errors')
def get_errors():
    """Получение логов ошибок всех сервисов"""
    try:
        errors = {}
        base_path = Path(__file__).parent / "logs"
        
        for service_name in SERVICES:
            stderr_file = base_path / f"{service_name}_stderr.log"
            if stderr_file.exists():
                try:
                    with open(stderr_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            errors[service_name] = content[-500:]  # Последние 500 символов
                except Exception as e:
                    errors[service_name] = f"Ошибка чтения: {e}"
        
        return jsonify({"errors": errors})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    """Обработка чат-сообщений с использованием памяти"""
    try:
        data = request.json
        message = data.get('message', '')
        discord_id = data.get('discord_id', 'unknown_user')
        
        # Получаем контекст из сервиса памяти
        memory_url = service_manager.get_service_url('memory_service')
        context = ""
        
        if memory_url:
            try:
                memory_response = requests.get(
                    f"{memory_url}/memory/context/{discord_id}",
                    params={"message": message},
                    timeout=5
                )
                if memory_response.status_code == 200:
                    context = memory_response.json().get('context', '')
            except Exception as e:
                logger.warning(f"Memory service unavailable: {e}")
        
        # Отправляем запрос к LLM сервису с контекстом
        llm_url = service_manager.get_service_url('llm_service')
        if llm_url:
            response = requests.post(
                f"{llm_url}/chat",
                json={
                    "message": message,
                    "context": context,
                    "max_length": 100,
                    "temperature": 0.7
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '')
                emotion = result.get('emotion', 'neutral')
                
                # Сохраняем диалог в памяти
                if memory_url:
                    try:
                        requests.post(
                            f"{memory_url}/memory/conversation",
                            json={
                                "discord_id": discord_id,
                                "messages": [
                                    {"role": "user", "message": message},
                                    {"role": "vtuber", "message": ai_response, "emotion": emotion}
                                ]
                            },
                            timeout=5
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save to memory: {e}")
                
                # Отправляем эмоцию аватар-сервису
                avatar_url = service_manager.get_service_url('avatar_service')
                if avatar_url:
                    try:
                        requests.post(
                            f"{avatar_url}/emotion",
                            json={"emotion": emotion},
                            timeout=5
                        )
                    except:
                        pass  # Не критично если аватар недоступен
                
                return jsonify(result)
            else:
                return jsonify({"error": "LLM service unavailable"}), 503
        else:
            return jsonify({"error": "LLM service not configured"}), 503
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/stats', methods=['GET'])
def memory_stats():
    """Получение статистики памяти"""
    try:
        memory_url = service_manager.get_service_url('memory_service')
        if memory_url:
            response = requests.get(f"{memory_url}/memory/stats", timeout=5)
            if response.status_code == 200:
                return jsonify(response.json())
        return jsonify({"error": "Memory service unavailable"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/memory/history/<discord_id>', methods=['GET'])
def get_user_history(discord_id):
    """Получение истории пользователя"""
    try:
        memory_url = service_manager.get_service_url('memory_service')
        if memory_url:
            limit = request.args.get('limit', 20)
            response = requests.get(
                f"{memory_url}/memory/history/{discord_id}",
                params={"limit": limit},
                timeout=5
            )
            if response.status_code == 200:
                return jsonify(response.json())
        return jsonify({"error": "Memory service unavailable"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/memory/profile/<discord_id>', methods=['GET'])
def get_user_profile(discord_id):
    """Получение профиля пользователя"""
    try:
        memory_url = service_manager.get_service_url('memory_service')
        if memory_url:
            response = requests.get(f"{memory_url}/memory/profile/{discord_id}", timeout=5)
            if response.status_code == 200:
                return jsonify(response.json())
            elif response.status_code == 404:
                return jsonify({"error": "Profile not found"}), 404
        return jsonify({"error": "Memory service unavailable"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/memory/viewer')
def memory_viewer():
    """Страница просмотра памяти"""
    return render_template('memory_viewer.html')

@app.route('/api/memory/users', methods=['GET'])
def get_memory_users():
    """Получение списка всех пользователей с сохраненными диалогами"""
    try:
        memory_url = service_manager.get_service_url('memory_service')
        if memory_url:
            response = requests.get(f"{memory_url}/memory/users", timeout=5)
            if response.status_code == 200:
                return jsonify(response.json())
        return jsonify({"error": "Memory service unavailable"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/memory')
def memory_redirect():
    """Редирект на страницу просмотра памяти"""
    return redirect('/memory/viewer')

@app.route('/admin')
def admin_redirect():
    """Редирект на основную страницу администрирования"""
    return redirect('/')

@app.route('/dashboard')
def dashboard_redirect():
    """Альтернативный редирект на основную страницу"""
    return redirect('/')

@socketio.on('connect')
def handle_connect():
    """Обработка подключения WebSocket"""
    logger.info('Client connected')
    emit('status', service_manager.get_all_services_status())

@socketio.on('disconnect')
def handle_disconnect():
    """Обработка отключения WebSocket"""
    logger.info('Client disconnected')

@socketio.on('chat_message')
def handle_chat_message(data):
    """Обработка чат-сообщений через WebSocket"""
    try:
        message = data.get('message', '')
        logger.info(f"Received message: {message}")
        
        # Здесь можно добавить обработку сообщения
        # Пока просто отправляем эхо
        emit('chat_response', {
            'response': f"Получил ваше сообщение: {message}",
            'emotion': 'neutral'
        })
        
    except Exception as e:
        logger.error(f"WebSocket chat error: {e}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    logger.info("Starting VTuber AI Main Server...")
    
    # Проверяем доступность сервисов при запуске
    status = service_manager.get_all_services_status()
    for service_name, service_status in status.items():
        if service_status['running']:
            logger.info(f"✅ {service_name} is running")
        else:
            logger.warning(f"⚠️  {service_name} is not available")
    
    socketio.run(
        app,
        host=SERVICES['main_server']['host'],
        port=SERVICES['main_server']['port'],
        debug=False,
        allow_unsafe_werkzeug=True
    )
