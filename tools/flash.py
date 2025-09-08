import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import base64
import time
from flask_cors import CORS
import json
import uuid
import threading

app = Flask(__name__)
CORS(app)

# Глобальные переменные для хранения данных
processing_progress = 0
preprocessed_data = {}
current_session_id = None

def preprocess_image(image, relief_strength=0.05, session_id=None):
    global preprocessed_data

    if session_id is None:
        session_id = str(uuid.uuid4())

    # Конвертируем в numpy array
    img = np.array(image)

    # Применяем фильтр масляной живописи
    oil_img = cv2.xphoto.oilPainting(img, 3, 1)

    # Конвертируем в оттенки серого для анализа
    gray = cv2.cvtColor(oil_img, cv2.COLOR_RGB2GRAY)

    # Вычисляем градиенты для определения направления мазков
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # Создаем карту высот на основе градиентов
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    orientation = np.arctan2(sobel_y, sobel_x)

    # Нормализуем величину градиента
    magnitude = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude

    # Создаем базовую карту высот на основе градиентов
    height_map = np.zeros_like(magnitude, dtype=np.float32)

    # Добавляем случайный шум для естественности
    noise = np.random.rand(*height_map.shape) * 0.01 * relief_strength
    height_map += noise

    # Определяем размеры изображения
    h, w = img.shape[:2]

    # Разбиваем изображение на сегменты (квадраты разного размера)
    segment_sizes = [16, 24, 32, 48]  # Различные размеры сегментов
    segment_probability = 0.6  # Вероятность применения рельефа к сегменту

    for size in segment_sizes:
        for y in range(0, h, size):
            for x in range(0, w, size):
                # Случайно решаем, применять ли рельеф к этому сегменту
                if np.random.random() > segment_probability:
                    continue

                # Определяем границы сегмента
                y_end = min(y + size, h)
                x_end = min(x + size, w)

                # Случайно выбираем тип рельефа для сегмента
                relief_type = np.random.choice(['gradient', 'bump', 'ridge', 'random'])

                if relief_type == 'gradient':
                    # Градиент от одной стороны к другой
                    direction = np.random.choice(['horizontal', 'vertical'])

                    if direction == 'horizontal':
                        # Градиент слева направо или справа налево
                        reverse = np.random.random() > 0.5
                        gradient = np.linspace(0, 1, x_end - x)
                        if reverse:
                            gradient = 1 - gradient
                        # Применяем градиент ко всем строкам сегмента
                        for i in range(y, y_end):
                            height_map[i, x:x_end] += relief_strength * 0.3 * gradient * magnitude[i, x:x_end]

                    else:  # vertical
                        # Градиент сверху вниз или снизу вверх
                        reverse = np.random.random() > 0.5
                        gradient = np.linspace(0, 1, y_end - y)
                        if reverse:
                            gradient = 1 - gradient
                        # Применяем градиент ко всем столбцам сегмента
                        for j in range(x, x_end):
                            height_map[y:y_end, j] += relief_strength * 0.3 * gradient * magnitude[y:y_end, j]

                elif relief_type == 'bump':
                    # Выпуклый бугорок в центре сегмента
                    center_y = y + (y_end - y) // 2
                    center_x = x + (x_end - x) // 2

                    # Создаем выпуклость с гауссовым распределением
                    yy, xx = np.ogrid[y:y_end, x:x_end]
                    distance = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
                    max_distance = np.sqrt((size/2)**2 + (size/2)**2)
                    bump = np.exp(-(distance**2) / (2 * (size/4)**2))

                    height_map[y:y_end, x:x_end] += relief_strength * 0.4 * bump * magnitude[y:y_end, x:x_end]

                elif relief_type == 'ridge':
                    # Гребень через центр сегмента
                    direction = np.random.choice(['horizontal', 'vertical'])

                    if direction == 'horizontal':
                        center_y = y + (y_end - y) // 2
                        ridge_width = size // 4

                        for i in range(y, y_end):
                            distance = abs(i - center_y)
                            if distance < ridge_width:
                                ridge = 1 - (distance / ridge_width)
                                height_map[i, x:x_end] += relief_strength * 0.3 * ridge * magnitude[i, x:x_end]

                    else:  # vertical
                        center_x = x + (x_end - x) // 2
                        ridge_width = size // 4

                        for j in range(x, x_end):
                            distance = abs(j - center_x)
                            if distance < ridge_width:
                                ridge = 1 - (distance / ridge_width)
                                height_map[y:y_end, j] += relief_strength * 0.3 * ridge * magnitude[y:y_end, j]

                elif relief_type == 'random':
                    # Случайные вариации высоты в сегменте
                    random_heights = np.random.rand(y_end - y, x_end - x) * 0.2 * relief_strength
                    height_map[y:y_end, x:x_end] += random_heights * magnitude[y:y_end, x:x_end]

    # Добавляем волнообразную текстуру, имитирующую мазки кисти
    for y in range(h):
        for x in range(w):
            wave = np.sin(x * 0.05 + y * 0.03) * np.cos(y * 0.04 - x * 0.02)
            height_map[y, x] += relief_strength * 0.1 * wave * magnitude[y, x]

    # Размываем карту высот для плавности
    height_map = cv2.GaussianBlur(height_map, (5, 5), 0)

    # Нормализуем карту высот
    if height_map.max() > 0:
        height_map = height_map / height_map.max() * relief_strength

    # Рассчитываем нормали на основе карты высот
    sobel_x = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)

    # Нормализуем градиенты
    norm_magnitude = np.sqrt(sobel_x**2 + sobel_y**2 + 1e-6)
    normal_x = sobel_x / norm_magnitude
    normal_y = sobel_y / norm_magnitude
    normal_z = 1.0 / np.sqrt(normal_x**2 + normal_y**2 + 1)

    # Сохраняем предварительно обработанные данные
    preprocessed_data[session_id] = {
        'original_img': img,  # Исходное изображение для текстуры
        'oil_img': oil_img,   # Сглаженное для совместимости
        'height_map': height_map,
        'normal_x': normal_x,
        'normal_y': normal_y,
        'normal_z': normal_z,
        'img_shape': img.shape,
        'relief_strength': relief_strength
    }

    return session_id
def render_lighting(preprocessed_data, light_position):
    # Используем исходное изображение для текстуры (сохраняем детали)
    texture_img = preprocessed_data.get('original_img', preprocessed_data['oil_img'])
    normal_x = preprocessed_data['normal_x']
    normal_y = preprocessed_data['normal_y']
    normal_z = preprocessed_data['normal_z']
    img_shape = preprocessed_data['img_shape']

    # Плавное появление и исчезание освещения
    if light_position <= 30:
        intensity_factor = light_position / 30  # Плавное появление от 0 до 30
    elif light_position >= 170 and light_position <= 200:
        intensity_factor = (200 - light_position) / 30  # Плавное исчезание от 170 до 200
    elif light_position > 200:
        intensity_factor = 0.0  # Полное исчезание после 200
    else:
        intensity_factor = 1.0  # Полная интенсивность от 30 до 170

    # Рассчитываем направление света
    light_angle = light_position * 1.8  # 0-200 -> 0-360
    light_rad = np.radians(light_angle)

    # Вектор направления света
    light_dir = np.array([np.cos(light_rad), np.sin(light_rad), 0.7])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Рассчитываем интенсивность отражения (модель Ламберта)
    reflection = (normal_x * light_dir[0] +
                  normal_y * light_dir[1] +
                  normal_z * light_dir[2])

    # Применяем порог и регулируем силу эффекта
    reflection = np.clip(reflection, 0, 1)

    # Усиливаем эффект рельефа
    reflection = np.power(reflection, 2)

    # Применяем плавное появление/исчезание
    reflection *= intensity_factor

    # Создаем блики (теплый оттенок)
    highlight = np.zeros_like(texture_img, dtype=np.float32)
    highlight[:, :, 0] = 0.1 * reflection  # Синий канал
    highlight[:, :, 1] = 0.3 * reflection  # Зеленый канал
    highlight[:, :, 2] = 0.6 * reflection  # Красный канал

    # Добавляем основной блик от источника света
    center_x = int(img_shape[1] * light_position / 200)
    center_y = int(img_shape[0] * 0.3)

    # Создаем градиент для основного блика
    y, x = np.ogrid[:img_shape[0], :img_shape[1]]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt((img_shape[1])**2 + (img_shape[0])**2)
    main_highlight = np.exp(-(distance**2) / (2 * (max_distance/6)**2))

    # Применяем плавное появление/исчезание к основному блику
    main_highlight *= intensity_factor

    # Добавляем основной блик
    highlight[:, :, 0] += 0.1 * main_highlight
    highlight[:, :, 1] += 0.3 * main_highlight
    highlight[:, :, 2] += 0.6 * main_highlight

    # Накладываем блики на исходное изображение (сохраняем детали)
    result = texture_img.astype(np.float32) + 100 * highlight
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)

@app.route('/upload', methods=['POST'])
def upload_image():
    global processing_progress, current_session_id
    try:
        # Сбрасываем прогресс
        processing_progress = 0

        # Получаем файл и параметры
        file = request.files['image']
        relief_strength = float(request.form.get('relief_strength', 0.05))

        # Загружаем изображение и конвертируем в RGB
        image = Image.open(file.stream).convert('RGB')

        # Предварительная обработка
        processing_progress = 50
        session_id = preprocess_image(image, relief_strength)
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
        processed_image = render_lighting(preprocessed_data[session_id], light_position)

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
        <title>Oil Painting Effect</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .controls { margin-bottom: 20px; }
            .control-group { margin-bottom: 10px; }
            label { display: inline-block; width: 150px; }
            input[type="range"] { width: 300px; }
            #result-container { margin-top: 20px; }
            #processed-image { max-width: 100%; box-shadow: 0 0 10px rgba(0,0,0,0.3); }
            
            /* Стили для прогресс-бара */
            .progress-container {
                width: 100%;
                background-color: #f3f3f3;
                border-radius: 5px;
                margin: 10px 0;
                display: none;
            }
            
            .progress-bar {
                width: 0%;
                height: 20px;
                background-color: #4caf50;
                border-radius: 5px;
                text-align: center;
                line-height: 20px;
                color: white;
                transition: width 0.3s;
            }
            
            .button {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 5px;
            }
            
            .animation-controls {
                margin-top: 20px;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            
            .session-info {
                margin-top: 10px;
                padding: 10px;
                background-color: #e9e9e9;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Oil Painting Effect</h1>
        <form id="upload-form" enctype="multipart/form-data">
            <div class="control-group">
                <label for="image">Select Image:</label>
                <input type="file" id="image" name="image" accept="image/*">
            </div>
            <div class="control-group">
                <label for="relief_strength">Relief Strength (0.0-0.1):</label>
                <input type="range" id="relief_strength" name="relief_strength" min="0.0" max="0.1" step="0.01" value="0.05">
                <span id="relief-strength-value">0.05</span>
            </div>
            <button type="button" id="upload-btn" class="button">Upload and Preprocess</button>
        </form>
        
        <div id="session-info" class="session-info" style="display: none;">
            <h3>Session Info</h3>
            <p>Session ID: <span id="session-id"></span></p>
            <button type="button" id="cleanup-btn" class="button">Cleanup Session</button>
        </div>
        
        <div id="render-controls" style="display: none;">
            <div class="control-group">
                <label for="light_position">Light Position (0-200):</label>
                <input type="range" id="light_position" name="light_position" min="0" max="200" value="100">
                <span id="light-position-value">100</span>
            </div>
            <button type="button" id="render-btn" class="button">Render</button>
            
            <div class="animation-controls">
                <h3>Animation</h3>
                <div class="control-group">
                    <label for="animation-speed">Animation Speed:</label>
                    <input type="range" id="animation-speed" name="animation-speed" min="1" max="10" value="5">
                    <span id="animation-speed-value">5</span>
                </div>
                <button type="button" id="animate-btn" class="button">View Animation</button>
                <button type="button" id="stop-animation-btn" class="button" style="display: none;">Stop Animation</button>
                
                <div class="control-group">
                    <label for="animation-progress">Animation Progress:</label>
                    <progress id="animation-progress" value="0" max="100"></progress>
                    <span id="animation-progress-value">0%</span>
                </div>
            </div>
        </div>
        
        <!-- Контейнер для прогресс-бара -->
        <div id="progress-container" class="progress-container">
            <div id="progress-bar" class="progress-bar">0%</div>
        </div>
        
        <div id="result-container" style="display: none;">
            <h2>Processed Image</h2>
            <img id="processed-image">
        </div>
        
        <script>
            $(document).ready(function() {
                let animationInterval = null;
                let isAnimating = false;
                let currentSessionId = null;
                
                // Обновляем значения ползунков
                $('#light_position').on('input', function() {
                    $('#light-position-value').text($(this).val());
                });
                
                $('#relief_strength').on('input', function() {
                    $('#relief-strength-value').text($(this).val());
                });
                
                $('#animation-speed').on('input', function() {
                    $('#animation-speed-value').text($(this).val());
                });
                
                // Функция для обновления прогресс-бара
                function updateProgress() {
                    $.get('/progress', function(data) {
                        var progress = data.progress;
                        $('#progress-bar').css('width', progress + '%').text(progress + '%');
                        
                        // Если обработка еще не завершена, продолжаем опрашивать сервер
                        if (progress < 100) {
                            setTimeout(updateProgress, 100);
                        } else {
                            // Скрываем прогресс-бар через 1 секунду после завершения
                            setTimeout(function() {
                                $('#progress-container').fadeOut();
                                $('#render-controls').show();
                            }, 1000);
                        }
                    });
                }
                
                // Загрузка и предварительная обработка изображения
                $('#upload-btn').click(function() {
                    // Показываем прогресс-бар
                    $('#progress-container').show();
                    $('#progress-bar').css('width', '0%').text('0%');
                    
                    var formData = new FormData($('#upload-form')[0]);
                    
                    // Отправляем запрос на обработку
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
                                // Сохраняем session ID
                                currentSessionId = data.session_id;
                                $('#session-id').text(currentSessionId);
                                $('#session-info').show();
                                
                                // Запускаем обновление прогресса
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
                    if (!currentSessionId) {
                        alert('No active session. Please upload an image first.');
                        return;
                    }
                    
                    // Отправляем запрос на рендеринг
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
                
                // Автоматический рендеринг при изменении ползунка света
                $('#light_position').change(function() {
                    if (currentSessionId) {
                        var light_position = $('#light_position').val();
                        renderFrame(light_position);
                    }
                });
                
                // Функции для управления анимацией
                function startAnimation() {
                    if (isAnimating) return;
                    
                    isAnimating = true;
                    $('#animate-btn').hide();
                    $('#stop-animation-btn').show();
                    
                    var positions = [];
                    for (var i = 0; i <= 200; i += 5) {
                        positions.push(i);
                    }
                    
                    var speed = $('#animation-speed').val();
                    var delay = 110 - speed * 10; // От 100ms до 10ms
                    var totalFrames = positions.length;
                    var currentFrame = 0;
                    
                    animationInterval = setInterval(function() {
                        if (currentFrame >= totalFrames) {
                            stopAnimation()
                            return
                        }
                        
                        var pos = positions[currentFrame];
                        renderFrame(pos, false);
                        
                        // Обновляем прогресс анимации
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
                    
                    // Сбрасываем прогресс анимации
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