import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFilter
import io
import base64
import time
from flask_cors import CORS
import json
import uuid
import threading
import math
import random

app = Flask(__name__)
CORS(app)

# Глобальные переменные для хранения данных
processing_progress = 0
preprocessed_data = {}
current_session_id = None

def preprocess_image(image, flare_intensity=1.0, session_id=None):
    global preprocessed_data

    if session_id is None:
        session_id = str(uuid.uuid4())

    # Конвертируем в numpy array
    img = np.array(image)

    # Сохраняем оригинальное изображение
    preprocessed_data[session_id] = {
        'original_img': img,
        'flare_intensity': flare_intensity,
        'img_shape': img.shape
    }

    return session_id

def generate_professional_lens_flare(preprocessed_data, light_position):
    # Получаем данные из предобработки
    original_img = preprocessed_data['original_img']
    flare_intensity = preprocessed_data['flare_intensity']
    h, w, _ = original_img.shape

    # Создаем черное изображение для бликов в формате PIL для лучшего контроля
    flare_pil = Image.new('RGB', (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(flare_pil, 'RGBA')

    # Вычисляем позицию источника света на основе light_position (0-200)
    angle = (light_position / 200) * 2 * np.pi

    # Центр изображения
    center_x, center_y = w // 2, h // 2

    # Позиция источника света (за пределами изображения)
    source_distance = max(w, h) * 0.7
    source_x = center_x + source_distance * np.cos(angle)
    source_y = center_y + source_distance * np.sin(angle)

    # Основные параметры эффекта
    flare_elements = []

    # 1. ОСНОВНЫЕ ЛУЧИ (главный элемент эффекта)
    num_main_rays = 8
    ray_length = max(w, h) * 0.6
    ray_width = 15

    for i in range(num_main_rays):
        ray_angle = angle + (i * 2 * math.pi / num_main_rays)
        end_x = source_x + math.cos(ray_angle) * ray_length
        end_y = source_y + math.sin(ray_angle) * ray_length

        # Создаем луч с градиентом
        steps = 30
        for j in range(steps):
            t = j / steps
            current_x = source_x + (end_x - source_x) * t
            current_y = source_y + (end_y - source_y) * t
            current_width = ray_width * (1 - t * 0.8)
            current_intensity = 0.7 * (1 - t)

            # Цвет луча (от теплого к холодному)
            if t < 0.3:
                color = (255, 200, 100, int(200 * current_intensity))  # Теплый
            elif t < 0.6:
                color = (200, 220, 255, int(180 * current_intensity))  # Холодный
            else:
                color = (150, 180, 255, int(150 * current_intensity))  # Синий

            draw.ellipse([
                current_x - current_width/2,
                current_y - current_width/2,
                current_x + current_width/2,
                current_y + current_width/2
            ], fill=color)

    # 2. ОПТИЧЕСКИЕ КОЛЬЦА (характерные для объективов)
    num_rings = 5
    for i in range(num_rings):
        t = 0.2 + 0.6 * (i / (num_rings - 1))
        ring_x = center_x + (source_x - center_x) * (1 - t)
        ring_y = center_y + (source_y - center_y) * (1 - t)

        # Размер и интенсивность колец
        ring_size = 20 + 80 * (1 - abs(t - 0.5) * 2)
        ring_thickness = 3 + 8 * (1 - abs(t - 0.5) * 2)
        intensity = 0.8 * (1 - abs(t - 0.5) * 2)

        # Цвета колец (радужный эффект)
        colors = [
            (255, 100, 100, int(200 * intensity)),  # Красный
            (255, 200, 100, int(180 * intensity)),  # Оранжевый
            (255, 255, 100, int(160 * intensity)),  # Желтый
            (100, 255, 100, int(140 * intensity)),  # Зеленый
            (100, 200, 255, int(120 * intensity))   # Синий
        ]

        color = colors[i % len(colors)]

        # Рисуем кольцо
        for r in range(int(ring_thickness)):
            current_size = ring_size - r
            draw.ellipse([
                ring_x - current_size,
                ring_y - current_size,
                ring_x + current_size,
                ring_y + current_size
            ], outline=color, width=1)

    # 3. ОСНОВНОЙ ИСТОЧНИК СВЕТА (яркое ядро)
    core_size = 35
    # Ядро источника
    draw.ellipse([
        source_x - core_size,
        source_y - core_size,
        source_x + core_size,
        source_y + core_size
    ], fill=(255, 255, 220, 255))

    # Внешнее свечение
    for i in range(3):
        glow_size = core_size + 10 + i * 8
        alpha = 200 - i * 60
        draw.ellipse([
            source_x - glow_size,
            source_y - glow_size,
            source_x + glow_size,
            source_y + glow_size
        ], outline=(255, 240, 180, alpha), width=2)

    # 4. ВТОРИЧНЫЕ ЛУЧИ (менее яркие)
    num_secondary_rays = 16
    secondary_ray_length = max(w, h) * 0.4
    for i in range(num_secondary_rays):
        ray_angle = angle + (i * 2 * math.pi / num_secondary_rays) + 0.1
        end_x = source_x + math.cos(ray_angle) * secondary_ray_length
        end_y = source_y + math.sin(ray_angle) * secondary_ray_length

        steps = 20
        for j in range(steps):
            t = j / steps
            current_x = source_x + (end_x - source_x) * t
            current_y = source_y + (end_y - source_y) * t
            current_width = 5 * (1 - t)
            current_intensity = 0.4 * (1 - t)

            color = (200, 220, 255, int(150 * current_intensity))

            draw.ellipse([
                current_x - current_width/2,
                current_y - current_width/2,
                current_x + current_width/2,
                current_y + current_width/2
            ], fill=color)

    # 5. ДИФРАКЦИОННЫЕ ЭФФЕКТЫ (мелкие детали)
    num_diffraction = 50
    for i in range(num_diffraction):
        t = random.uniform(0.1, 0.9)
        diff_x = center_x + (source_x - center_x) * (1 - t) + random.randint(-50, 50)
        diff_y = center_y + (source_y - center_y) * (1 - t) + random.randint(-50, 50)

        size = random.randint(2, 8)
        intensity = random.uniform(0.3, 0.7)

        # Случайный цвет с преобладанием синих и фиолетовых оттенков
        colors = [
            (180, 180, 255, int(200 * intensity)),  # Голубой
            (200, 150, 255, int(180 * intensity)),  # Фиолетовый
            (150, 200, 255, int(160 * intensity))   # Синий
        ]

        color = random.choice(colors)

        draw.ellipse([
            diff_x - size,
            diff_y - size,
            diff_x + size,
            diff_y + size
        ], fill=color)

    # Конвертируем обратно в numpy array
    flare_np = np.array(flare_pil.convert('RGB')).astype(np.float32) / 255.0

    # Применяем размытие для естественного вида
    flare_blurred = cv2.GaussianBlur(flare_np, (0, 0), 3)

    # Комбинируем с оригинальным изображением (режим экран)
    result = original_img.astype(np.float32) / 255.0
    screen_effect = 1 - (1 - result) * (1 - flare_blurred)

    # Усиливаем эффект согласно настройке интенсивности
    result = cv2.addWeighted(screen_effect, 0.8, flare_blurred, flare_intensity * 0.6, 0)

    # Обеспечиваем корректные границы
    result = np.clip(result, 0, 1) * 255
    result = result.astype(np.uint8)

    return Image.fromarray(result)

@app.route('/upload', methods=['POST'])
def upload_image():
    global processing_progress, current_session_id
    try:
        # Сбрасываем прогресс
        processing_progress = 0

        # Получаем файл и параметры
        file = request.files['image']
        flare_intensity = float(request.form.get('flare_intensity', 1.0))

        # Загружаем изображение и конвертируем в RGB
        image = Image.open(file.stream).convert('RGB')

        # Предварительная обработка
        processing_progress = 50
        session_id = preprocess_image(image, flare_intensity)
        current_session_id = session_id
        processing_progress = 100

        return jsonify({'status': 'success', 'message': 'Image preprocessed successfully', 'session_id': session_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/render', methods=['POST'])
def render_image():
    try:
        # Получаем параметры
        data = request.get_json()
        light_position = int(data.get('light_position', 100))
        session_id = data.get('session_id', current_session_id)

        if session_id not in preprocessed_data:
            return jsonify({'error': 'No preprocessed image found for this session. Please upload an image first.'}), 400

        # Рендерим изображение с заданным положением света
        processed_image = generate_professional_lens_flare(preprocessed_data[session_id], light_position)

        # Конвертируем в base64 для ответа
        img_io = io.BytesIO()
        processed_image.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        img_str = base64.b64encode(img_io.getvalue()).decode('ascii')

        return jsonify({'image': img_str})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup():
    try:
        data = request.get_json()
        session_id = data.get('session_id')

        if session_id and session_id in preprocessed_data:
            del preprocessed_data[session_id]
            return jsonify({'status': 'success', 'message': f'Session {session_id} cleaned up'})
        else:
            return jsonify({'error': 'Session not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/progress')
def get_progress():
    return jsonify({'progress': processing_progress})

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Professional Lens Flare Effect</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #0a0a1a; color: #e0e0ff; }
            .container { max-width: 1000px; margin: 0 auto; background: #1a1a2a; padding: 20px; border-radius: 15px; box-shadow: 0 0 25px rgba(0, 100, 255, 0.2); }
            .controls { margin-bottom: 20px; padding: 20px; background: #2a2a3a; border-radius: 10px; border: 1px solid #3a3a5a; }
            .control-group { margin-bottom: 15px; display: flex; align-items: center; }
            label { display: inline-block; width: 200px; font-weight: bold; color: #4fc3f7; }
            input[type="range"] { width: 300px; margin-right: 10px; background: #3a3a5a; }
            .value-display { width: 50px; text-align: center; font-weight: bold; color: #4fc3f7; }
            #result-container { margin-top: 20px; text-align: center; }
            #processed-image { max-width: 100%; box-shadow: 0 0 30px rgba(79, 195, 247, 0.4); border-radius: 8px; }
            
            .progress-container {
                width: 100%;
                background-color: #2a2a3a;
                border-radius: 8px;
                margin: 15px 0;
                display: none;
            }
            
            .progress-bar {
                width: 0%;
                height: 20px;
                background: linear-gradient(90deg, #4fc3f7, #01579b);
                border-radius: 8px;
                text-align: center;
                line-height: 20px;
                color: #000;
                font-weight: bold;
                transition: width 0.3s;
            }
            
            .button {
                background: linear-gradient(45deg, #01579b, #4fc3f7);
                border: none;
                color: white;
                padding: 12px 25px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                font-weight: bold;
                margin: 8px 5px;
                cursor: pointer;
                border-radius: 6px;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            .button:hover {
                transform: translateY(-3px);
                box-shadow: 0 5px 20px rgba(79, 195, 247, 0.4);
            }
            
            .animation-controls {
                margin-top: 20px;
                padding: 20px;
                background: linear-gradient(135deg, #2a2a3a, #1a1a2a);
                border-radius: 10px;
                border: 1px solid #4fc3f7;
            }
            
            .session-info {
                margin-top: 15px;
                padding: 15px;
                background: #2a2a3a;
                border-radius: 8px;
                border-left: 4px solid #4fc3f7;
            }
            
            h1 { color: #4fc3f7; text-align: center; text-shadow: 0 0 15px rgba(79, 195, 247, 0.7); margin-bottom: 10px; }
            h2 { color: #29b6f6; border-bottom: 2px solid #4fc3f7; padding-bottom: 8px; margin-top: 0; }
            h3 { color: #4fc3f7; margin-top: 0; }
            
            .description { text-align: center; color: #a0a0cc; margin-bottom: 25px; }
            
            input[type="file"] {
                background: #3a3a5a;
                color: #e0e0ff;
                padding: 10px;
                border-radius: 6px;
                border: 1px solid #4fc3f7;
                width: 300px;
            }
            
            progress {
                background: #3a3a5a;
                border: 1px solid #4fc3f7;
                height: 20px;
                border-radius: 6px;
            }
            
            progress::-webkit-progress-bar { background: #3a3a5a; border-radius: 6px; }
            progress::-webkit-progress-value { background: linear-gradient(90deg, #4fc3f7, #01579b); border-radius: 6px; }
            
            .feature-list {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin: 20px 0;
            }
            
            .feature {
                width: 48%;
                background: #2a2a3a;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 8px;
                border-left: 3px solid #4fc3f7;
            }
            
            .feature h4 { color: #4fc3f7; margin-top: 0; }
            .feature p { color: #c0c0e0; margin-bottom: 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌞 PROFESSIONAL LENS FLARE EFFECT 🌞</h1>
            <p class="description">Создавайте реалистичные оптические блики с выраженными лучами и кольцами, как в профессиональной фотографии</p>
            
            <div class="feature-list">
                <div class="feature">
                    <h4>🔦 Четкие лучи</h4>
                    <p>Яркие выраженные лучи, расходящиеся от источника света</p>
                </div>
                <div class="feature">
                    <h4>⭕ Оптические кольца</h4>
                    <p>Характерные круги, образующиеся в системе линз объектива</p>
                </div>
                <div class="feature">
                    <h4>🎯 Физическая точность</h4>
                    <p>Эффект, соответствующий реальным оптическим явлениям</p>
                </div>
                <div class="feature">
                    <h4>🌈 Реалистичное свечение</h4>
                    <p>Естественное распределение света и цветовых аберраций</p>
                </div>
            </div>
            
            <form id="upload-form" enctype="multipart/form-data">
                <div class="controls">
                    <h2>📁 Загрузка изображения</h2>
                    <div class="control-group">
                        <label for="image">Выберите изображение:</label>
                        <input type="file" id="image" name="image" accept="image/*" required>
                    </div>
                    <div class="control-group">
                        <label for="flare_intensity">Интенсивность эффекта:</label>
                        <input type="range" id="flare_intensity" name="flare_intensity" min="0.5" max="2.0" step="0.1" value="1.0">
                        <span id="flare-intensity-value" class="value-display">1.0</span>
                    </div>
                    <button type="button" id="upload-btn" class="button">🚀 Загрузить и обработать</button>
                </div>
            </form>
            
            <div id="session-info" class="session-info" style="display: none;">
                <h3>📊 Информация о сессии</h3>
                <p>ID сессии: <span id="session-id" style="color: #4fc3f7;"></span></p>
                <button type="button" id="cleanup-btn" class="button">🧹 Очистить сессию</button>
            </div>
            
            <div id="render-controls" style="display: none;">
                <div class="controls">
                    <h2>🎛️ Управление эффектом</h2>
                    <div class="control-group">
                        <label for="light_position">Положение источника:</label>
                        <input type="range" id="light_position" name="light_position" min="0" max="200" value="100">
                        <span id="light-position-value" class="value-display">100</span>
                    </div>
                    <button type="button" id="render-btn" class="button">🖼️ Применить эффект</button>
                    
                    <div class="animation-controls">
                        <h3>🎬 Анимация эффекта</h3>
                        <div class="control-group">
                            <label for="animation-speed">Скорость анимации:</label>
                            <input type="range" id="animation-speed" name="animation-speed" min="1" max="10" value="5">
                            <span id="animation-speed-value" class="value-display">5</span>
                        </div>
                        <button type="button" id="animate-btn" class="button">▶️ Запустить анимацию</button>
                        <button type="button" id="stop-animation-btn" class="button" style="display: none;">⏹️ Остановить</button>
                        
                        <div class="control-group">
                            <label for="animation-progress">Прогресс анимации:</label>
                            <progress id="animation-progress" value="0" max="100" style="width: 300px;"></progress>
                            <span id="animation-progress-value" class="value-display">0%</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="progress-container" class="progress-container">
                <div id="progress-bar" class="progress-bar">0%</div>
            </div>
            
            <div id="result-container" style="display: none;">
                <h2>🎨 Результат обработки</h2>
                <img id="processed-image">
            </div>
        </div>
        
        <script>
            $(document).ready(function() {
                let animationInterval = null;
                let isAnimating = false;
                let currentSessionId = null;
                
                // Обновляем значения ползунков
                $('#light_position').on('input', function() {
                    $('#light-position-value').text($(this).val());
                    if (currentSessionId) {
                        renderFrame($(this).val(), false);
                    }
                });
                
                $('#flare_intensity').on('input', function() {
                    $('#flare-intensity-value').text($(this).val());
                });
                
                $('#animation-speed').on('input', function() {
                    $('#animation-speed-value').text($(this).val());
                });
                
                // Функция для обновления прогресс-бара
                function updateProgress() {
                    $.get('/progress', function(data) {
                        var progress = data.progress;
                        $('#progress-bar').css('width', progress + '%').text(progress + '%');
                        
                        if (progress < 100) {
                            setTimeout(updateProgress, 100);
                        } else {
                            setTimeout(function() {
                                $('#progress-container').fadeOut();
                                $('#render-controls').show();
                                // Автоматически рендерим первое изображение
                                renderFrame($('#light_position').val(), true);
                            }, 1000);
                        }
                    });
                }
                
                // Загрузка и предварительная обработка изображения
                $('#upload-btn').click(function() {
                    $('#progress-container').show();
                    $('#progress-bar').css('width', '0%').text('0%');
                    
                    var formData = new FormData($('#upload-form')[0]);
                    
                    $.ajax({
                        url: '/upload',
                        type: 'POST',
                        data: formData,
                        contentType: false,
                        processData: false,
                        success: function(data) {
                            if (data.error) {
                                alert('Error: ' + data.error);
                                $('#progress-container').hide();
                            } else {
                                currentSessionId = data.session_id;
                                $('#session-id').text(currentSessionId);
                                $('#session-info').show();
                                updateProgress();
                            }
                        },
                        error: function() {
                            alert('Server error occurred');
                            $('#progress-container').hide();
                        }
                    });
                });
                
                // Очистка сессии
                $('#cleanup-btn').click(function() {
                    if (!currentSessionId) return;
                    
                    $.ajax({
                        url: '/cleanup',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({
                            session_id: currentSessionId
                        }),
                        success: function(data) {
                            if (data.error) {
                                alert('Error: ' + data.error);
                            } else {
                                alert('Session cleaned up successfully');
                                currentSessionId = null;
                                $('#session-id').text('');
                                $('#session-info').hide();
                                $('#render-controls').hide();
                                $('#result-container').hide();
                            }
                        },
                        error: function() {
                            alert('Server error occurred');
                        }
                    });
                });
                
                // Рендеринг изображения
                function renderFrame(light_position, updateUI = true) {
                    if (!currentSessionId) return;
                    
                    $.ajax({
                        url: '/render',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({
                            light_position: light_position,
                            session_id: currentSessionId
                        }),
                        success: function(data) {
                            if (data.error) {
                                alert('Error: ' + data.error);
                                if (isAnimating) stopAnimation();
                            } else {
                                $('#processed-image').attr('src', 'data:image/jpeg;base64,' + data.image);
                                $('#result-container').show();
                                
                                if (updateUI) {
                                    $('#light_position').val(light_position);
                                    $('#light-position-value').text(light_position);
                                }
                            }
                        },
                        error: function() {
                            alert('Server error occurred');
                            if (isAnimating) stopAnimation();
                        }
                    });
                }
                
                // Рендеринг по кнопке
                $('#render-btn').click(function() {
                    var light_position = $('#light_position').val();
                    renderFrame(light_position);
                });
                
                // Функции для управления анимацией
                function startAnimation() {
                    if (isAnimating) return;
                    
                    isAnimating = true;
                    $('#animate-btn').hide();
                    $('#stop-animation-btn').show();
                    
                    var positions = [];
                    for (var i = 0; i <= 200; i += 2) {
                        positions.push(i);
                    }
                    
                    var speed = $('#animation-speed').val();
                    var delay = 110 - speed * 10;
                    var totalFrames = positions.length;
                    var currentFrame = 0;
                    
                    animationInterval = setInterval(function() {
                        if (currentFrame >= totalFrames) {
                            stopAnimation()
                            return
                        }
                        
                        var pos = positions[currentFrame];
                        renderFrame(pos, false);
                        
                        var progress = Math.round((currentFrame / totalFrames) * 100);
                        $('#animation-progress').val(progress);
                        $('#animation-progress-value').text(progress + '%');
                        
                        currentFrame++;
                    }, delay);
                }
                
                function stopAnimation() {
                    if (!isAnimating) return;
                    
                    isAnimating = false;
                    clearInterval(animationInterval);
                    $('#animate-btn').show();
                    $('#stop-animation-btn').hide();
                    
                    $('#animation-progress').val(0);
                    $('#animation-progress-value').text('0%');
                }
                
                // Запуск анимации
                $('#animate-btn').click(function() {
                    startAnimation();
                });
                
                // Остановка анимации
                $('#stop-animation-btn').click(function() {
                    stopAnimation();
                });
            });
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)