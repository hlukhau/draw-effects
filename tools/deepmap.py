import numpy as np
import matplotlib
# Используем неинтерактивный бэкенд для избежания конфликтов с Flask
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import io
import base64
import tempfile
import os
from flask import Flask, render_template_string, request

app = Flask(__name__)

# HTML шаблон в виде строки
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>3D Depth Map Generator</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #333; }
        form { margin: 20px 0; }
        .result { margin-top: 30px; }
        .loading { display: none; color: blue; }
        .error { color: red; margin-top: 20px; }
    </style>
    <script>
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').style.display = 'none';
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>Загрузите изображение для создания 3D-анимации</h1>
        <form action="/" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
            <input type="file" name="file" accept="image/*" required>
            <input type="submit" value="Создать анимацию">
        </form>
        
        <div id="loading" class="loading">Обработка изображения, пожалуйста подождите...</div>
        
        {% if animation_url %}
        <div class="result">
            <h2>Результат:</h2>
            <img src="{{ animation_url }}" alt="3D анимация">
            <p><a href="{{ animation_url }}" download="3d_animation.gif">Скачать анимацию</a></p>
        </div>
        {% endif %}
        
        <div id="error" class="error" style="display: {% if error %}block{% else %}none{% endif %};">
            {% if error %}<p>{{ error }}</p>{% endif %}
        </div>
    </div>
</body>
</html>
'''

def create_depth_map(img):
    """Создает синтетическую карту глубины"""
    img_gray = img.convert('L')
    depth_map = np.array(img_gray) / 255.0

    # Добавляем градиент для эффекта глубины
    h, w = depth_map.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    # Центральная точка для создания эффекта выпуклости/вогнутости
    center_x, center_y = w/2, h/2
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)

    # Комбинируем яркость с радиальным градиентом
    depth_map = 0.6 * depth_map + 0.4 * (1 - dist_from_center / max_dist)

    return depth_map

def create_rotation_animation(depth_map):
    """Создает анимацию вращения 3D-поверхности"""
    # Создаем фигуру и оси
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    # Уменьшаем разрешение для производительности
    step = max(1, depth_map.shape[0] // 80)
    depth_map_small = depth_map[::step, ::step]

    x = np.linspace(0, 1, depth_map_small.shape[1])
    y = np.linspace(0, 1, depth_map_small.shape[0])
    x, y = np.meshgrid(x, y)

    # Настраиваем график
    ax.set_axis_off()
    ax.grid(False)
    ax.view_init(elev=30, azim=0)

    def update(frame):
        ax.clear()
        ax.set_axis_off()
        ax.grid(False)
        surf = ax.plot_surface(x, y, depth_map_small, cmap='viridis',
                               edgecolor='none', alpha=0.8)
        ax.view_init(elev=30, azim=frame)
        ax.set_zlim(0, 1)
        return ax,

    # Создаем анимацию с меньшим количеством кадров для производительности
    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 20),
                        interval=100, blit=False, repeat=True)

    # Создаем временный файл для сохранения анимации
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Сохраняем анимацию во временный файл
        ani.save(temp_path, writer='pillow', fps=8, dpi=80)

        # Читаем файл обратно в буфер
        with open(temp_path, 'rb') as f:
            buf = io.BytesIO(f.read())

        buf.seek(0)
        return buf
    finally:
        # Всегда удаляем временный файл
        try:
            os.unlink(temp_path)
        except:
            pass
        plt.close(fig)

@app.route('/', methods=['GET', 'POST'])
def index():
    animation_url = None
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = "Файл не загружен"
            return render_template_string(HTML_TEMPLATE, error=error)

        file = request.files['file']
        if file.filename == '':
            error = "Файл не выбран"
            return render_template_string(HTML_TEMPLATE, error=error)

        try:
            # Открываем изображение
            img = Image.open(file.stream)

            # Ограничиваем размер для производительности
            max_size = (300, 300)
            img.thumbnail(max_size)

            # Создаем карту глубины
            depth_map = create_depth_map(img)

            # Создаем анимацию
            animation = create_rotation_animation(depth_map)

            # Кодируем в base64 для отображения на странице
            animation_data = base64.b64encode(animation.getvalue()).decode('utf-8')
            animation_url = f"data:image/gif;base64,{animation_data}"

        except Exception as e:
            error = f"Ошибка обработки изображения: {str(e)}"
            return render_template_string(HTML_TEMPLATE, error=error)

    return render_template_string(HTML_TEMPLATE, animation_url=animation_url, error=error)

if __name__ == '__main__':
    app.run(debug=True, threaded=False)  # Отключаем многопоточность для избежания конфликтов