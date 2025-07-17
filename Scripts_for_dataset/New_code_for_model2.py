import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import numpy as np
import os
import imagehash

# Список имен классов
class_names = [
    "argali", "Asian-badger", "badger", "beaver", "brown-bear", "capercaillie", "crane", "field-lark", "fox",
    "gold-eagle", "Groundhog-baibak", "hare", "heron", "hog", "kedrovka", "lynx", "mallard", "marten", "mericanmink",
    "moose", "murtiger", "otter", "polecat", "raccoon-dog", "sub-deer", "swan", "ular", "white-tailed-eagle", "wolf",
    "woodpecker"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detection_threshold = 0.8  # порог для детектора
classification_threshold = 0.8  # порог для классификатора

# --- 1. Загрузка детекционной модели (Faster R-CNN) ---
detection_weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
detection_model = fasterrcnn_resnet50_fpn(weights=detection_weights)
detection_model.to(device)
detection_model.eval()

# --- 2. Загрузка классификационной модели ---
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
classification_model = torchvision.models.efficientnet_b0(weights=weights)
num_features = classification_model.classifier[1].in_features
# Если у вас 15 классов, оставьте 15, если их больше, то измените число выходов
classification_model.classifier[1] = torch.nn.Linear(num_features, len(class_names))
classification_model.load_state_dict(torch.load('best_efficientnet.pth', map_location=device))
classification_model.to(device)
classification_model.eval()

# --- 3. Трансформации для классификации ---
classification_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- 4. Подготовка папок для сохранения кропов ---
output_root = "test_video2"  # корневая папка для сохранения кропов
if not os.path.exists(output_root):
    os.makedirs(output_root)

# Для каждого класса создадим подпапку и набор хешей для проверки дубликатов
hash_sets = {}
for cls in class_names:
    class_folder = os.path.join(output_root, cls.replace(" ", "_"))
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    hash_sets[cls] = set()

# Порог для сравнения хешей (можно экспериментировать)
hash_threshold = 5


def is_duplicate(crop_pil, saved_hashes):
    current_hash = imagehash.phash(crop_pil)
    for saved_hash in saved_hashes:
        if abs(current_hash - saved_hash) < hash_threshold:
            return True, current_hash
    return False, current_hash


input_video_path = "test_vid2.mp4"  # путь к исходному видео
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Не удалось открыть видео.")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Настраиваем VideoWriter для сохранения обработанного видео
output_video_path = os.path.join(output_root, "processed_video.avi")
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
crop_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Преобразуем кадр из BGR в RGB и создаем тензор
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = F.to_tensor(frame_rgb).to(device)

    # Детекция объектов на кадре
    with torch.no_grad():
        detections = detection_model([frame_tensor])[0]

    boxes = detections['boxes']
    scores = detections['scores']

    # Вычисляем временную метку кадра
    current_time = frame_count / fps
    minutes = int(current_time // 60)
    seconds = int(current_time % 60)

    # Обрабатываем каждую детекцию
    for i in range(len(scores)):
        if scores[i] < detection_threshold:
            continue

        box = boxes[i].detach().cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        # Защита от некорректного бокса
        if x2 <= x1 or y2 <= y1:
            continue

        # Извлекаем кроп (область детекции)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Преобразуем кроп в формат для классификации
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_tensor = classification_transform(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = classification_model(crop_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            conf = conf.item()
            pred = pred.item()

        if conf < classification_threshold:
            continue

        # Получаем название класса по индексу
        class_label = class_names[pred]

        # Проверка на дубликат
        duplicate, current_hash = is_duplicate(crop_pil, hash_sets[class_label])
        if duplicate:
            continue
        hash_sets[class_label].add(current_hash)

        # Формат имени: минута_секунда_точность_класс.jpg
        crop_filename = f"{minutes:02d}_{seconds:02d}_{conf * 100:.1f}_{class_label}.jpg"
        class_folder = os.path.join(output_root, class_label.replace(" ", "_"))
        crop_filepath = os.path.join(class_folder, crop_filename)
        cv2.imwrite(crop_filepath, crop)
        crop_count += 1

        # Отрисовка прямоугольника и подписи на кадре
        label_text = f"{class_label}: {conf * 100:.1f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Запись обработанного кадра в видео
    out_video.write(frame)

    # Можно отображать кадры для контроля
    cv2.imshow("Processed Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()

print(f"Обработка завершена. Сохранено {crop_count} кропов в папке '{output_root}'.")
print(f"Обработанное видео сохранено по пути: {output_video_path}")
