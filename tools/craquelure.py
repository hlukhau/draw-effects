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
import random
import math

app = Flask(__name__)
CORS(app)

# Глобальные переменные для хранения данных
processing_progress = 0
preprocessed_data = {}
current_session_id = None

def generate_craquelure_pattern(width, height, density=0.5, complexity=3):
    """Генерирует аутентичный кракелюр (трещины) как на старых картинах"""
    # Создаем прозрачное изображение для трещин
    crackle_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(crackle_img)

    # Основные трещины - начинаются от краев
    main_cracks = []
    num_main_cracks = int(density * 20) + 10

    for _ in range(num_main_cracks):
        # Выбираем случайную точку на границе
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            start_x = random.randint(0, width - 1)
            start_y = 0
        elif side == 'bottom':
            start_x = random.randint(0, width - 1)
            start_y = height - 1
        elif side == 'left':
            start_x = 0
            start_y = random.randint(0, height - 1)
        else:  # right
            start_x = width - 1
            start_y = random.randint(0, height - 1)

        main_cracks.append((start_x, start_y))

    # Создаем сеть трещин
    all_cracks = main_cracks.copy()

    for i in range(complexity):
        new_cracks = []
        for crack in all_cracks:
            # Создаем ответвления от существующих трещин
            if random.random() < 0.7:  # Вероятность ответвления
                angle = random.uniform(0, 2 * math.pi)
                length = random.randint(20, min(width, height) // (i + 2))

                end_x = int(crack[0] + length * math.cos(angle))
                end_y = int(crack[1] + length * math.sin(angle))

                # Обеспечиваем, чтобы трещина оставалась в пределах изображения
                end_x = max(0, min(width - 1, end_x))
                end_y = max(0, min(height - 1, end_y))

                # Рисуем трещину
                thickness = max(1, random.randint(1, 3 - i))
                color = (0, 0, 0, random.randint(150, 220))  # Полупрозрачный черный

                draw.line([crack, (end_x, end_y)], fill=color, width=thickness)

                new_cracks.append((end_x, end_y))

        all_cracks.extend(new_cracks)

    # Добавляем микротрещины
    num_micro_cracks = int(density * 100)
    for _ in range(num_micro_cracks):
        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)

        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(5, 15)

        end_x = int(start_x + length * math.cos(angle))
        end_y = int(start_y + length * math.sin(angle))

        end_x = max(0, min(width - 1, end_x))
        end_y = max(0, min(height - 1, end_y))

        thickness = random.choice([1, 1, 1, 2])  # В основном тонкие трещины
        color = (0, 0, 0, random.randint(100, 180))

        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=thickness)

    return crackle_img

def apply_authentic_aging_effect(image, aging_level=0.5, session_id=None):
    global preprocessed_data

    if session_id is None:
        session_id = str(uuid.uuid4())

    # Конвертируем в numpy array
    img = np.array(image)
    height, width = img.shape[:2]

    # Сохраняем оригинальное изображение
    original_img = img.copy()

    # 1. Выцветание цветов - более тонкое
    fade_factor = 0.8 + (aging_level * 0.2)  # 0.8-1.0
    faded_img = (img.astype(np.float32) * fade_factor).astype(np.uint8)

    # 2. Добавляем теплый желтоватый оттенок (патина времени)
    sepia_tone = np.array([[[15, 25, 40]]], dtype=np.uint8) * aging_level
    aged_img = np.clip(faded_img.astype(np.int16) + sepia_tone, 0, 255).astype(np.uint8)

    # 3. Создаем аутентичный кракелюр
    crackle_density = 0.3 + (aging_level * 0.7)  # 0.3-1.0
    crackle_complexity = min(4, int(aging_level * 3) + 1)

    crackle_texture = generate_craquelure_pattern(width, height, crackle_density, crackle_complexity)
    crackle_array = np.array(crackle_texture)

    # 4. Легкое текстурирование - имитация поверхности холста
    canvas_texture = np.random.rand(height, width) * 15 * aging_level
    canvas_texture = np.stack([canvas_texture] * 3, axis=-1)
    textured_img = np.clip(aged_img.astype(np.int16) - canvas_texture.astype(np.int16), 0, 255).astype(np.uint8)

    # 5. Конвертируем обратно в PIL для наложения кракелюра
    base_image = Image.fromarray(textured_img)

    # 6. Накладываем кракелюр
    final_image = Image.alpha_composite(
        base_image.convert('RGBA'),
        crackle_texture
    ).convert('RGB')

    final_array = np.array(final_image)

    # 7. Добавляем легкое виньетирование
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_radius = np.sqrt(center_x**2 + center_y**2)

    vignette = 1 - 0.4 * aging_level * (radius / max_radius)**1.5
    vignette = np.clip(vignette, 0.6, 1)
    vignette = np.stack([vignette] * 3, axis=-1)

    final_array = (final_array.astype(np.float32) * vignette).astype(np.uint8)

    # 8. Легкое размытие для естественности
    if aging_level > 0.3:
        blur_amount = int(aging_level * 1.5)
        if blur_amount > 0:
            blurred = cv2.GaussianBlur(final_array, (blur_amount*2+1, blur_amount*2+1), 0)
            # Слегка смешиваем с размытой версией
            alpha = 0.2 * aging_level
            final_array = cv2.addWeighted(final_array, 1-alpha, blurred, alpha, 0)

    # Сохраняем предварительно обработанные данные
    preprocessed_data[session_id] = {
        'original_img': original_img,
        'aged_img': final_array,
        'crackle_texture': crackle_array,
        'aging_level': aging_level,
        'img_shape': original_img.shape
    }

    return session_id

def apply_aging_intensity(preprocessed_data, intensity):
    """Применяет интенсивность старения к предобработанному изображению"""
    original_img = preprocessed_data['original_img']
    aged_img = preprocessed_data['aged_img']
    base_aging_level = preprocessed_data['aging_level']

    # Корректируем интенсивность относительно базового уровня старения
    effective_intensity = min(1.5, intensity * base_aging_level * 1.2)

    if effective_intensity <= 1.0:
        # Плавный переход от оригинального к состаренному
        result = cv2.addWeighted(original_img, 1-effective_intensity, aged_img, effective_intensity, 0)
    else:
        # Для высокой интенсивности - дополнительное выцветание
        extra_fade = 0.9 - (effective_intensity - 1.0) * 0.1
        result = (aged_img.astype(np.float32) * extra_fade).astype(np.uint8)

    return Image.fromarray(result)

@app.route('/upload', methods=['POST'])
def upload_image():
    global processing_progress, current_session_id
    try:
        # Сбрасываем прогресс
        processing_progress = 0

        # Получаем файл и параметры
        file = request.files['image']
        aging_level = float(request.form.get('aging_level', 0.5))

        # Загружаем изображение и конвертируем в RGB
        image = Image.open(file.stream).convert('RGB')

        # Предварительная обработка
        processing_progress = 50
        session_id = apply_authentic_aging_effect(image, aging_level)
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
        aging_intensity = float(data.get('aging_intensity', 1.0))
        session_id = data.get('session_id', current_session_id)

        if session_id not in preprocessed_data:
            return jsonify({'error': 'No preprocessed image found for this session. Please upload an image first.'}), 400

        # Применяем эффект старения с заданной интенсивностью
        processed_image = apply_aging_intensity(preprocessed_data[session_id], aging_intensity)

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
        <title>Authentic Painting Aging</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            body { font-family: 'Georgia', serif; margin: 0; padding: 20px; background: #f8f6f2; color: #5d4037; }
            .container { max-width: 1200px; margin: 0 auto; background: #fffaf0; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(93, 64, 55, 0.1); border: 2px solid #d7ccc8; }
            .controls { margin-bottom: 30px; padding: 25px; background: linear-gradient(135deg, #f5f5f5, #e8e8e8); border-radius: 12px; border-left: 5px solid #8d6e63; }
            .control-group { margin-bottom: 20px; display: flex; align-items: center; padding: 10px; background: rgba(255, 255, 255, 0.8); border-radius: 8px; }
            label { display: inline-block; width: 220px; font-weight: bold; color: #5d4037; font-size: 16px; }
            input[type="range"] { flex: 1; margin: 0 20px; height: 8px; background: #d7ccc8; border-radius: 4px; }
            input[type="range"]::-webkit-slider-thumb { appearance: none; width: 20px; height: 20px; background: #8d6e63; border-radius: 50%; cursor: pointer; }
            .value-display { width: 70px; text-align: center; font-weight: bold; color: #8d6e63; font-size: 16px; background: #f5f5f5; padding: 5px; border-radius: 5px; border: 1px solid #d7ccc8; }
            
            .progress-container {
                width: 100%;
                background: linear-gradient(135deg, #f5f5f5, #e8e8e8);
                border-radius: 10px;
                margin: 20px 0;
                display: none;
                border: 2px solid #d7ccc8;
                overflow: hidden;
            }
            
            .progress-bar {
                width: 0%;
                height: 30px;
                background: linear-gradient(90deg, #8d6e63, #a1887f);
                text-align: center;
                line-height: 30px;
                color: white;
                font-weight: bold;
                font-size: 14px;
                transition: width 0.8s ease-in-out;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
            }
            
            .button {
                background: linear-gradient(45deg, #8d6e63, #a1887f);
                border: none;
                color: white;
                padding: 15px 30px;
                text-align: center;
                font-size: 16px;
                margin: 12px 8px;
                cursor: pointer;
                border-radius: 30px;
                transition: all 0.3s ease;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
                box-shadow: 0 4px 15px rgba(141, 110, 99, 0.3);
            }
            
            .button:hover {
                background: linear-gradient(45deg, #6d4c41, #8d6e63);
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(141, 110, 99, 0.4);
            }
            
            .animation-controls {
                margin-top: 30px;
                padding: 25px;
                background: linear-gradient(135deg, #f3e5f5, #e8eaf6);
                border-radius: 12px;
                border-left: 5px solid #7e57c2;
            }
            
            .session-info {
                margin-top: 20px;
                padding: 20px;
                background: linear-gradient(135deg, #e8f5e8, #f1f8e9);
                border-radius: 12px;
                border-left: 5px solid #66bb6a;
            }
            
            .image-comparison {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-top: 30px;
            }
            
            .image-container {
                text-align: center;
                padding: 20px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border: 3px solid #d7ccc8;
                transition: transform 0.3s ease;
            }
            
            .image-container:hover {
                transform: translateY(-5px);
            }
            
            .image-container img {
                max-width: 100%;
                border: 2px solid #d7ccc8;
                border-radius: 8px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
                transition: all 0.3s ease;
            }
            
            .image-container img:hover {
                transform: scale(1.02);
                box-shadow: 0 8px 30px rgba(0,0,0,0.2);
            }
            
            .image-label {
                margin-top: 15px;
                font-weight: bold;
                color: #5d4037;
                font-size: 18px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            h1, h2, h3 {
                color: #5d4037;
                text-align: center;
                margin-bottom: 25px;
                text-transform: uppercase;
                letter-spacing: 2px;
            }
            
            h1 {
                font-size: 36px;
                margin-bottom: 10px;
                color: #8d6e63;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            
            .effect-description {
                text-align: center;
                color: #8d6e63;
                margin-bottom: 30px;
                font-style: italic;
                font-size: 18px;
                line-height: 1.6;
            }
            
            .info-text {
                text-align: center;
                color: #8d6e63;
                margin: 15px 0;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 Authentic Painting Aging</h1>
            <p class="effect-description">Transform your images into aged masterpieces with authentic craquelure patterns</p>
            
            <div class="controls">
                <h2>🖼️ Upload & Settings</h2>
                <form id="upload-form" enctype="multipart/form-data">
                    <div class="control-group">
                        <label for="image">Select Image:</label>
                        <input type="file" id="image" name="image" accept="image/*" style="flex: 2; padding: 8px; border: 2px solid #d7ccc8; border-radius: 5px;">
                    </div>
                    <div class="control-group">
                        <label for="aging_level">Craquelure Density (0.1-1.0):</label>
                        <input type="range" id="aging_level" name="aging_level" min="0.1" max="1.0" step="0.1" value="0.5">
                        <span id="aging-level-value" class="value-display">0.5</span>
                    </div>
                    <div style="text-align: center; margin-top: 20px;">
                        <button type="button" id="upload-btn" class="button">🎨 Apply Authentic Aging</button>
                    </div>
                </form>
                <p class="info-text">Higher values create more intricate crack patterns</p>
            </div>
            
            <div id="progress-container" class="progress-container">
                <div id="progress-bar" class="progress-bar">Creating masterpiece... 0%</div>
            </div>
            
            <div id="session-info" class="session-info" style="display: none;">
                <h3>📋 Session Information</h3>
                <div class="control-group">
                    <label>Session ID:</label>
                    <span id="session-id" style="flex: 1; font-family: monospace; background: #f5f5f5; padding: 10px; border-radius: 5px; border: 1px solid #d7ccc8; color: #8d6e63;"></span>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <button type="button" id="cleanup-btn" class="button">🧹 Cleanup Session</button>
                </div>
            </div>
            
            <div id="render-controls" style="display: none;">
                <h2>⚙️ Aging Controls</h2>
                <div class="control-group">
                    <label for="aging_intensity">Aging Intensity (0.0-1.5):</label>
                    <input type="range" id="aging_intensity" name="aging_intensity" min="0.0" max="1.5" step="0.05" value="1.0">
                    <span id="aging-intensity-value" class="value-display">1.0</span>
                </div>
                <p class="info-text">Adjust to see different stages of aging</p>
                
                <div class="animation-controls">
                    <h3>✨ Aging Animation</h3>
                    <div class="control-group">
                        <label for="animation-speed">Animation Speed:</label>
                        <input type="range" id="animation-speed" name="animation-speed" min="1" max="10" value="5">
                        <span id="animation-speed-value" class="value-display">5</span>
                    </div>
                    <div style="text-align: center;">
                        <button type="button" id="animate-btn" class="button">▶️ View Aging Process</button>
                        <button type="button" id="stop-animation-btn" class="button" style="display: none;">⏹️ Stop Animation</button>
                    </div>
                    
                    <div class="control-group">
                        <label for="animation-progress">Progress:</label>
                        <progress id="animation-progress" value="0" max="100" style="flex: 1; height: 20px;"></progress>
                        <span id="animation-progress-value" class="value-display">0%</span>
                    </div>
                </div>
            </div>
            
            <div id="result-container" style="display: none;">
                <h2>🖼️ Result Preview</h2>
                <div class="image-comparison">
                    <div class="image-container">
                        <div class="image-label">Original Artwork</div>
                        <img id="original-image">
                    </div>
                    <div class="image-container">
                        <div class="image-label">Aged Masterpiece</div>
                        <img id="processed-image">
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            $(document).ready(function() {
                let animationInterval = null;
                let isAnimating = false;
                let currentSessionId = null;
                
                // Обновляем значения ползунков
                $('#aging_level').on('input', function() {
                    $('#aging-level-value').text($(this).val());
                });
                
                $('#aging_intensity').on('input', function() {
                    $('#aging-intensity-value').text($(this).val());
                    if (currentSessionId) {
                        renderFrame($(this).val());
                    }
                });
                
                $('#animation-speed').on('input', function() {
                    $('#animation-speed-value').text($(this).val());
                });
                
                // Функция для обновления прогресс-бара
                function updateProgress() {
                    $.get('/progress', function(data) {
                        var progress = data.progress;
                        $('#progress-bar').css('width', progress + '%').text('Creating masterpiece... ' + progress + '%');
                        
                        if (progress < 100) {
                            setTimeout(updateProgress, 100);
                        } else {
                            setTimeout(function() {
                                $('#progress-container').fadeOut();
                                $('#render-controls').show();
                                $('#result-container').show();
                            }, 1000);
                        }
                    });
                }
                
                // Загрузка и обработка изображения
                $('#upload-btn').click(function() {
                    $('#progress-container').show();
                    $('#progress-bar').css('width', '0%').text('Creating masterpiece... 0%');
                    
                    var formData = new FormData($('#upload-form')[0]);
                    
                    // Сохраняем оригинальное изображение для preview
                    var fileInput = $('#image')[0];
                    if (fileInput.files.length > 0) {
                        var reader = new FileReader();
                        reader.onload = function(e) {
                            $('#original-image').attr('src', e.target.result);
                        };
                        reader.readAsDataURL(fileInput.files[0]);
                    }
                    
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
                    
                    if (confirm('Are you sure you want to cleanup this session?')) {
                        $.ajax({
                            url: '/cleanup',
                            type: 'POST',
                            contentType: 'application/json',
                            data: JSON.stringify({ session_id: currentSessionId }),
                            success: function(data) {
                                currentSessionId = null;
                                $('#session-id').text('');
                                $('#session-info').hide();
                                $('#render-controls').hide();
                                $('#result-container').hide();
                                $('#progress-container').hide();
                            },
                            error: function() {
                                alert('Server error occurred');
                            }
                        });
                    }
                });
                
                // Рендеринг изображения
                function renderFrame(aging_intensity, updateUI = true) {
                    if (!currentSessionId) return;
                    
                    $.ajax({
                        url: '/render',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({
                            aging_intensity: aging_intensity,
                            session_id: currentSessionId
                        }),
                        success: function(data) {
                            if (data.error) {
                                alert('Error: ' + data.error);
                                if (isAnimating) stopAnimation();
                            } else {
                                $('#processed-image').attr('src', 'data:image/jpeg;base64,' + data.image);
                                if (updateUI) {
                                    $('#aging_intensity').val(aging_intensity);
                                    $('#aging-intensity-value').text(aging_intensity);
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
                $('#render-btn')?.click(function() {
                    var intensity = $('#aging_intensity').val();
                    renderFrame(intensity);
                });
                
                // Автоматический рендеринг при изменении ползунка
                $('#aging_intensity').change(function() {
                    if (currentSessionId) {
                        var intensity = $('#aging_intensity').val();
                        renderFrame(intensity);
                    }
                });
                
                // Функции для управления анимацией
                function startAnimation() {
                    if (isAnimating) return;
                    
                    isAnimating = true;
                    $('#animate-btn').hide();
                    $('#stop-animation-btn').show();
                    
                    var intensities = [];
                    for (var i = 0; i <= 15; i++) {
                        intensities.push(i * 0.1);
                    }
                    
                    var speed = $('#animation-speed').val();
                    var delay = 120 - speed * 10;
                    var totalFrames = intensities.length;
                    var currentFrame = 0;
                    
                    animationInterval = setInterval(function() {
                        if (currentFrame >= totalFrames) {
                            stopAnimation();
                            return;
                        }
                        
                        var intensity = intensities[currentFrame];
                        renderFrame(intensity, false);
                        
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
                $('#animate-btn').click(startAnimation);
                $('#stop-animation-btn').click(stopAnimation);
            });
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)