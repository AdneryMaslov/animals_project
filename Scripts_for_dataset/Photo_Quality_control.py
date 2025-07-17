import os
import cv2
import shutil
import numpy as np
from skimage.restoration import estimate_sigma
from skimage import img_as_float
from PIL import Image

input_dir = r"data_filtered"  # здесь находятся папки с классами
bad_dir = r"data_filtered_bad"  # сюда будут перемещаться «плохие» изображения

if not os.path.exists(bad_dir):
    os.makedirs(bad_dir)

# Новые пороговые значения
min_width = 100       # минимальная ширина изображения (в пикселях)
min_height = 100      # минимальная высота изображения
blur_threshold = 80   # порог для variance of Laplacian (ниже – изображение размыто)
brightness_low = 30   # нижний порог средней яркости (0-255)
brightness_high = 220 # верхний порог средней яркости
noise_threshold = 0.1  # порог для оценки шума (в нормированных значениях)

def is_low_quality(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось открыть {image_path}")
            return True  # если не удалось открыть – считаем плохим

        height, width = img.shape[:2]
        # Проверка разрешения
        if width < min_width or height < min_height:
            return True

        # Переводим в оттенки серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Оценка размытости: variance of Laplacian
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < blur_threshold:
            return True

        # Оценка яркости: средняя интенсивность
        mean_brightness = np.mean(gray)
        if mean_brightness < brightness_low or mean_brightness > brightness_high:
            return True

        # Оценка шума: перевод в формат float
        img_float = img_as_float(gray)
        # Если изображение в оттенках серого (2D), можно вызвать estimate_sigma без дополнительных параметров
        sigma_est = np.mean(estimate_sigma(img_float))
        if sigma_est > noise_threshold:
            return True

        return False
    except Exception as e:
        print(f"Ошибка обработки {image_path}: {e}")
        return True

# Обход всех классов (папок) в input_dir
for class_name in os.listdir(input_dir):
    class_input_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_input_path):
        continue

    # Создаём аналогичную структуру в bad_dir
    class_bad_path = os.path.join(bad_dir, class_name)
    if not os.path.exists(class_bad_path):
        os.makedirs(class_bad_path)

    # Перебираем изображения в классе
    for filename in os.listdir(class_input_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        image_path = os.path.join(class_input_path, filename)
        if is_low_quality(image_path):
            # Если изображение низкокачественное, перемещаем его в папку bad
            dest_path = os.path.join(class_bad_path, filename)
            print(f"Перемещаем {image_path} -> {dest_path}")
            shutil.move(image_path, dest_path)

print("Проверка качества изображений завершена.")
