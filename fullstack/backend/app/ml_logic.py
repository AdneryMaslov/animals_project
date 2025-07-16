import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tensorflow as tf

from config import settings


# ===================================================================
# ==            ЛОГИКА ДЛЯ КЛАССИФИКАТОРА (PyTorch)               ==
# ===================================================================

def get_classifier_model() -> torch.nn.Module:
    """
    Загружает модель ResNet50 для КЛАССИФИКАЦИИ, обученную на PyTorch.
    """
    model = models.resnet50(weights=None)
    num_classes = len(settings.SPECIES_CLASSES)

    # Воспроизводим точную архитектуру финального слоя из ноутбука
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, num_classes)
    )

    try:
        state_dict = torch.load(settings.CLASSIFIER_WEIGHTS, map_location=torch.device('cpu'))
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('resnet.'):
                # Отбрасываем 'resnet.' (7 символов)
                new_key = key[7:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict)
        print("Модель классификатора (PyTorch) успешно загружена.")
    except Exception as e:
        print(
            f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить веса классификатора из {settings.CLASSIFIER_WEIGHTS}. Ошибка: {e}")
        raise e

    model.eval()
    return model


# Трансформации для изображений, которые подаются в классификатор
classifier_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_species(model: torch.nn.Module, img_path: str) -> str:
    """
    Принимает путь к изображению и модель, возвращает предсказанное название вида.
    """
    try:
        input_image = Image.open(img_path).convert('RGB')
        input_tensor = classifier_preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            _, predicted_idx = torch.max(probabilities, 0)

        return settings.SPECIES_CLASSES[predicted_idx.item()]
    except Exception as e:
        print(f"Ошибка при классификации изображения {img_path}: {e}")
        return "Unknown"


# ===================================================================
# ==            ЛОГИКА ДЛЯ ЭМБЕДДЕРА (TensorFlow)                  ==
# ===================================================================

def get_embedder_model() -> tf.keras.Model:
    """Загружает модель ResNet50 для ИЗВЛЕЧЕНИЯ ЭМБЕДДИНГОВ на TensorFlow."""
    try:
        model = tf.keras.applications.ResNet50(
            weights=settings.EMBEDDER_WEIGHTS,
            include_top=False,
            pooling='avg'
        )
        print("Модель эмбеддера (TensorFlow) успешно загружена.")
        return model
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить веса эмбеддера из {settings.EMBEDDER_WEIGHTS}. Ошибка: {e}")
        raise e


def extract_embedding(model: tf.keras.Model, img_path: str) -> list[float]:
    """Извлекает вектор признаков (эмбеддинг) из файла изображения."""
    try:
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        features = model.predict(img_array, verbose=0)
        return [float(x) for x in features.flatten()]

    except Exception as e:
        print(f"Ошибка при извлечении эмбеддинга из {img_path}: {e}")
        return []


# ===================================================================
# ==            ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ                            ==
# ===================================================================

def cosine_similarity(a: list, b: list) -> float:
    if not a or not b: return 0.0
    a_np, b_np = np.array(a), np.array(b)
    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot_product / (norm_a * norm_b)


def find_most_similar_passports(new_embedding: list[float], existing_passports: list[tuple[int, list[float]]]) -> list[
    tuple[int, float]]:
    if not new_embedding: return []
    similarities = []
    for pass_id, pass_embedding in existing_passports:
        sim = cosine_similarity(new_embedding, pass_embedding)
        similarities.append((pass_id, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

