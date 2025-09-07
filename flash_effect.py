import numpy as np
import cv2
from PIL import Image
import io
import base64
import uuid


class FlashEffect:
    """
    Автономный класс для создания эффекта светового рельефа с анимацией.
    Основан на логике из tools/flash.py
    """
    
    def __init__(self):
        self.preprocessed_data = {}
        self.processing_progress = 0
    
    def preprocess_image(self, image, relief_strength=0.05, session_id=None):
        """
        Предварительная обработка изображения для создания карты высот и нормалей
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Конвертируем в numpy array
        img = np.array(image)

        # Применяем фильтр масляной живописи (альтернативная реализация без opencv-contrib)
        # Используем билатеральный фильтр для имитации эффекта масляной живописи
        oil_img = cv2.bilateralFilter(img, 15, 50, 50)
        # Дополнительное размытие для более мягкого эффекта
        oil_img = cv2.medianBlur(oil_img, 5)

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
        self.preprocessed_data[session_id] = {
            'oil_img': oil_img,
            'height_map': height_map,
            'normal_x': normal_x,
            'normal_y': normal_y,
            'normal_z': normal_z,
            'img_shape': img.shape,
            'relief_strength': relief_strength
        }

        return session_id

    def render_lighting(self, session_id, light_position):
        """
        Рендерит кадр анимации с заданным положением света
        """
        if session_id not in self.preprocessed_data:
            raise ValueError(f"No preprocessed data found for session {session_id}")
        
        preprocessed_data = self.preprocessed_data[session_id]
        oil_img = preprocessed_data['oil_img']
        normal_x = preprocessed_data['normal_x']
        normal_y = preprocessed_data['normal_y']
        normal_z = preprocessed_data['normal_z']
        img_shape = preprocessed_data['img_shape']

        # Плавное появление и исчезание освещения
        if light_position <= 30:
            intensity_factor = light_position / 30  # Плавное появление от 0 до 30
        elif light_position >= 170:
            intensity_factor = (200 - light_position) / 30  # Плавное исчезание от 170 до 200
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
        highlight = np.zeros_like(oil_img, dtype=np.float32)
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

        # Накладываем блики на изображение
        result = oil_img.astype(np.float32) + 100 * highlight
        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result)

    def get_animation_frames(self, session_id, frame_step=1):
        """
        Генерирует все кадры анимации для одного цикла
        Возвращает список кадров как PIL Images
        """
        if session_id not in self.preprocessed_data:
            raise ValueError(f"No preprocessed data found for session {session_id}")
        
        frames = []
        positions = list(range(0, 201, frame_step))  # 0-200 с шагом frame_step
        
        for position in positions:
            frame = self.render_lighting(session_id, position)
            frames.append(frame)
        
        return frames, positions

    def cleanup_session(self, session_id):
        """
        Очищает предрасчитанные данные для указанной сессии
        """
        if session_id in self.preprocessed_data:
            del self.preprocessed_data[session_id]
            return True
        return False

    def cleanup_all_sessions(self):
        """
        Очищает все предрасчитанные данные
        """
        self.preprocessed_data.clear()
        self.processing_progress = 0

    def get_session_count(self):
        """
        Возвращает количество активных сессий
        """
        return len(self.preprocessed_data)

    def frame_to_base64(self, frame_image):
        """
        Конвертирует PIL Image в base64 строку
        """
        img_io = io.BytesIO()
        frame_image.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        img_str = base64.b64encode(img_io.getvalue()).decode('ascii')
        return img_str
