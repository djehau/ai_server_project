"""
API для сервиса долговременной памяти
Предоставляет REST endpoint'ы для работы с памятью
"""

from flask import Flask, request, jsonify
from memory_service import get_memory_service
import logging
from datetime import datetime
import uuid

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
logger = logging.getLogger(__name__)

# Получаем сервис памяти
memory = get_memory_service()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "memory_service"})

@app.route('/memory/save', methods=['POST'])
def save_message():
    """Сохраняет сообщение в память"""
    try:
        data = request.json
        
        discord_id = data.get('discord_id')
        role = data.get('role', 'user')
        message = data.get('message', '')
        emotion = data.get('emotion')
        context = data.get('context')
        session_id = data.get('session_id')
        
        if not discord_id or not message:
            return jsonify({"error": "discord_id и message обязательны"}), 400
        
        memory.save_message(
            discord_id=discord_id,
            role=role,
            message=message,
            emotion=emotion,
            context=context,
            session_id=session_id
        )
        
        return jsonify({"success": True, "message": "Сообщение сохранено"})
        
    except Exception as e:
        logger.error(f"Ошибка сохранения сообщения: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/conversation', methods=['POST'])
def save_conversation():
    """Сохраняет целый диалог"""
    try:
        data = request.json
        
        discord_id = data.get('discord_id')
        messages = data.get('messages', [])
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not discord_id or not messages:
            return jsonify({"error": "discord_id и messages обязательны"}), 400
        
        memory.save_conversation(
            discord_id=discord_id,
            messages=messages,
            session_id=session_id
        )
        
        return jsonify({"success": True, "session_id": session_id})
        
    except Exception as e:
        logger.error(f"Ошибка сохранения диалога: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/history/<discord_id>', methods=['GET'])
def get_history(discord_id):
    """Получает историю диалогов пользователя"""
    try:
        limit = int(request.args.get('limit', 20))
        
        history = memory.get_conversation_history(discord_id, limit)
        
        return jsonify({
            "discord_id": discord_id,
            "messages": history,
            "count": len(history)
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения истории: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/profile/<discord_id>', methods=['GET'])
def get_profile(discord_id):
    """Получает профиль пользователя"""
    try:
        profile = memory.get_user_profile(discord_id)
        
        if profile:
            return jsonify(profile)
        else:
            return jsonify({"error": "Профиль не найден"}), 404
            
    except Exception as e:
        logger.error(f"Ошибка получения профиля: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/profile/<discord_id>/preferences', methods=['PUT'])
def update_preferences(discord_id):
    """Обновляет предпочтения пользователя"""
    try:
        data = request.json
        preferences = data.get('preferences', {})
        
        memory.update_user_preferences(discord_id, preferences)
        
        return jsonify({"success": True, "message": "Предпочтения обновлены"})
        
    except Exception as e:
        logger.error(f"Ошибка обновления предпочтений: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/context/<discord_id>', methods=['GET'])
def get_context(discord_id):
    """Получает контекст для AI"""
    try:
        current_message = request.args.get('message', '')
        
        context = memory.get_context_for_ai(discord_id, current_message)
        
        return jsonify({
            "discord_id": discord_id,
            "context": context
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения контекста: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/stats', methods=['GET'])
def get_stats():
    """Получает статистику памяти"""
    try:
        stats = memory.get_memory_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/cleanup', methods=['POST'])
def cleanup_old_messages():
    """Очищает старые сообщения"""
    try:
        data = request.json
        days_old = data.get('days_old', 30)
        
        deleted_count = memory.clear_old_messages(days_old)
        
        return jsonify({
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Удалено {deleted_count} старых сообщений"
        })
        
    except Exception as e:
        logger.error(f"Ошибка очистки: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/users', methods=['GET'])
def get_users():
    """Получает список всех пользователей с сохраненными диалогами"""
    try:
        users = memory.get_all_users()
        return jsonify({
            "users": users,
            "count": len(users)
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения пользователей: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/history/<discord_id>/detailed', methods=['GET'])
def get_detailed_history(discord_id):
    """Получает детальную историю с ID для редактирования"""
    try:
        limit = int(request.args.get('limit', 50))
        
        history = memory.get_conversation_with_ids(discord_id, limit)
        
        return jsonify({
            "discord_id": discord_id,
            "messages": history,
            "count": len(history)
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения детальной истории: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/message/<int:message_id>', methods=['DELETE'])
def delete_message(message_id):
    """Удаляет сообщение"""
    try:
        success = memory.delete_message(message_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Сообщение {message_id} удалено"
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Сообщение {message_id} не найдено"
            }), 404
        
    except Exception as e:
        logger.error(f"Ошибка удаления сообщения: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/message/<int:message_id>', methods=['PUT'])
def edit_message(message_id):
    """Редактирует сообщение"""
    try:
        data = request.json
        new_message = data.get('message')
        new_emotion = data.get('emotion')
        
        if not new_message:
            return jsonify({"error": "Текст сообщения обязателен"}), 400
        
        success = memory.edit_message(message_id, new_message, new_emotion)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Сообщение {message_id} обновлено"
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Сообщение {message_id} не найдено"
            }), 404
        
    except Exception as e:
        logger.error(f"Ошибка редактирования сообщения: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/user/<discord_id>', methods=['DELETE'])
def delete_user(discord_id):
    """Удаляет все данные пользователя"""
    try:
        result = memory.delete_user_data(discord_id)
        
        if result.get('success'):
            return jsonify({
                "success": True,
                "message": f"Данные пользователя {discord_id} удалены",
                "deleted_messages": result.get('deleted_messages', 0),
                "deleted_profile": result.get('deleted_profile', False)
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', f"Пользователь {discord_id} не найден")
            }), 404
        
    except Exception as e:
        logger.error(f"Ошибка удаления пользователя: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=8086, debug=True)
