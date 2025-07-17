import os
import random
import cv2
from albumentations import (
    Compose, HorizontalFlip, RandomRotate90, ShiftScaleRotate, RandomBrightnessContrast,
    CLAHE, Blur, GaussNoise, OpticalDistortion, GridDistortion, HueSaturationValue,
    CoarseDropout, Normalize
)
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np


def augment_image(image, transform):
    augmented = transform(image=np.array(image))
    return augmented['image']


# Определяем трансформацию (агрессивная аугментация)
aggressive_transform = Compose([
    HorizontalFlip(p=0.5),
    RandomRotate90(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
    RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
    GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    Blur(blur_limit=7, p=0.3),
    CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5)
])


def aggressive_augment_dataset(input_dir, output_dir, target_count=700):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    class_list = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for cls in class_list:
        class_input_dir = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(class_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(images)
        print(f"Класс '{cls}': {current_count} исходных изображений.")

        class_output_dir = os.path.join(output_dir, cls)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        if current_count >= target_count:
            selected = random.sample(images, target_count)
            for img in selected:
                src = os.path.join(class_input_dir, img)
                dst = os.path.join(class_output_dir, img)
                cv2.imwrite(dst, cv2.imread(src))
            print(f"Класс '{cls}' сбалансирован (выбрано {target_count} изображений).")
        else:
            # Скопировать все исходные изображения
            for img in images:
                src = os.path.join(class_input_dir, img)
                dst = os.path.join(class_output_dir, img)
                cv2.imwrite(dst, cv2.imread(src))
            needed = target_count - current_count
            print(f"Класс '{cls}': недостаточно изображений. Генерация {needed} аугментированных изображений.")
            idx = 0
            while needed > 0:
                img_name = images[idx % current_count]
                img_path = os.path.join(class_input_dir, img_name)
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Ошибка открытия {img_path}: {e}")
                    idx += 1
                    continue
                augmented = augment_image(image, aggressive_transform)
                augmented_img = Image.fromarray(augmented)
                new_img_name = f"aug_{idx:04d}_{img_name}"
                augmented_img.save(os.path.join(class_output_dir, new_img_name))
                needed -= 1
                idx += 1
            print(f"Класс '{cls}' сбалансирован до {target_count} изображений.")

    print("Балансировка классов завершена.")


if __name__ == "__main__":
    input_dir = r"NewDataset/mammals"  # Папка с исходными изображениями
    output_dir = r"NewDataset_balanced/mammals"  # Папка для сбалансированного датасета
    aggressive_augment_dataset(input_dir, output_dir, target_count=1000)
