import os
import cv2
import numpy as np

# Папка с исходными изображениями (уже отфильтрованными)
input_folder = r"data_filtered"
# Папка, куда будут сохранены денойзенные изображения
output_folder = r"data_denoised"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Пороговые значения для яркости (0-255)
brightness_low = 50    # ниже – ночной режим
brightness_high = 205  # выше – очень ярко (снегопад, сильное солнце)

# Обходим все классы (папки) в input_folder
for class_name in os.listdir(input_folder):
    class_input_path = os.path.join(input_folder, class_name)
    if not os.path.isdir(class_input_path):
        continue

    # Создаем соответствующую папку для денойзенных изображений
    class_output_path = os.path.join(output_folder, class_name)
    if not os.path.exists(class_output_path):
        os.makedirs(class_output_path)

    for filename in os.listdir(class_input_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        image_path = os.path.join(class_input_path, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось открыть {image_path}")
            continue

        # Вычисляем среднюю яркость (по преобразованному в оттенки серого изображению)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if brightness < brightness_low:
            # Тёмные (ночные) изображения – шум, как правило, сильнее, увеличиваем параметры
            h = 15
            hColor = 15
        elif brightness > brightness_high:
            # Очень яркие изображения – шум может быть другого характера, уменьшаем немного параметры
            h = 8
            hColor = 8
        else:
            h = 10
            hColor = 10

        # Стандартные размеры окна (подбираются экспериментально)
        templateWindowSize = 7
        searchWindowSize = 21

        print(f"Обработка {filename}: brightness={brightness:.1f}, h={h}, hColor={hColor}")

        denoised = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)

        output_path = os.path.join(class_output_path, filename)
        cv2.imwrite(output_path, denoised)
        print(f"Сохранено: {output_path}")

print("Денойзинг завершён.")
