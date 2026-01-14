"""Программа для детекции и распознавания лиц на видео (Terminator-Style)"""

import cv2
import numpy as np
import os
import ast
from PIL import Image
from faces_video_config import *
from datetime import datetime

print('\nПривет! Я программа по распознаванию лиц на видео.\n')


# 1. Инициализация детектора лиц (используем каскад Хаара)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# 2. Создаём распознаватель лиц (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer.create()


# 3. Словарь для соответствия ID и имени. Заполняем из файла
with open(NAMES_FILE, 'r', encoding='utf-8') as f:
    # Читаем содержимое файла и превращаем строку в словарь
    names = ast.literal_eval(f.read())


# 4. Путь к датасету
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)


# 5.1 Функция для добавления лица в базу по фото с камеры
def add_face_to_dataset():
    """Добавления моего лица в базу по фото с веб-камеры"""
    face_id = 1  # ID для вашего лица
    count = 0

    cap = cv2.VideoCapture(0)  # Веб-камера
    print("Создание датасета. Смотрите в камеру...")
    print("Собираю 30 образцов вашего лица...")

    while count < 30:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Детектируем лица
        faces = face_cascade.detectMultiScale(gray, gray, **DETECTION_PARAMS['photo'])

        for (x, y, w, h) in faces:
            # Рисуем прямоугольник
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Сохраняем только лицо
            face_roi = gray[y:y + h, x:x + w]

            # Увеличиваем лицо до стандартного размера
            face_resized = cv2.resize(face_roi, (200, 200))

            # Сохраняем в датасет
            count += 1
            cv2.imwrite(f"{DATASET_PATH}/User.{face_id}.{count}.jpg", face_resized)
            print(f"Сохранён образец {count}/30")

        # Показываем процесс
        cv2.imshow('Добавление лица в базу', frame)

        # Задержка для сбора разных ракурсов
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Датасет создан!")

    # Обучаем модель
    train_recognizer()


# 5.2 Функция для добавления лиц в базу из готовых фото
def add_face_from_existing_photos(photo_folder: str, photo_id: int):  # Прикрутить в цикл по папкам!!!
    """
    Загружает готовые фото из папки вместо съёмки с камеры
    photo_folder - папка с фотографиями (jpg/png)
    """
    face_id = photo_id  # ID лица

    if not os.path.exists(photo_folder):
        print(f"Ошибка: папка '{photo_folder}' не найдена!")
        print(f"Создайте папку '{photo_folder}' и добавьте туда фотографии")
        return

    # Получаем список фото
    photo_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    photo_files = [f for f in os.listdir(photo_folder)
                   if os.path.splitext(f)[1].lower() in photo_extensions]

    if not photo_files:
        print(f"В папке '{photo_folder}' нет изображений!")
        return

    print(f"Найдено {len(photo_files)} фото. Обрабатываю...")

    count = 0
    for i, photo_file in enumerate(photo_files):
        photo_path = os.path.join(photo_folder, photo_file)

        try:
            # Загружаем фото
            img = cv2.imread(photo_path)
            if img is None:
                print(f"Не могу загрузить: {photo_file}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Детектируем лица на фото
            faces = face_cascade.detectMultiScale(gray, **DETECTION_PARAMS['photo'])

            if len(faces) == 0:
                print(f"На фото {photo_file} не найдено лиц")
                continue

            # Берём первое найденное лицо (предполагаем, что это вы)
            for (x, y, w, h) in faces[:1]:  # Берём только первое лицо
                # Сохраняем только лицо
                face_roi = gray[y:y + h, x:x + w]

                # Увеличиваем лицо до стандартного размера
                face_resized = cv2.resize(face_roi, (200, 200))

                # Сохраняем в датасет
                count += 1
                cv2.imwrite(f"{DATASET_PATH}/User.{face_id}.{count}.jpg", face_resized)
                print(f"Обработано фото {i + 1}/{len(photo_files)}: сохранено лицо {count}")

                # Показываем найденное лицо (опционально)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow(f"Найдено лицо в {photo_file}", img)
                cv2.waitKey(500)  # Показываем 0.5 секунды
                cv2.destroyWindow(f"Найдено лицо в {photo_file}")

        except Exception as e:
            print(f"Ошибка при обработке {photo_file}: {e}")
            continue

    cv2.destroyAllWindows()
    print(f"Датасет создан! Сохранено {count} образцов лица.")

    return count  # Возвращаем количество добавленных образцов


# 6. Функция обучения модели
def train_recognizer():
    """Функция обучения модели"""
    faces = []
    ids = []

    # Проверяем, есть ли файлы в датасете
    image_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.jpg')]

    if not image_files:
        print("Нет изображений в датасете!")
        return

    for image_name in image_files:
        try:
            # Извлекаем ID из имени файла (формат: User.id.number.jpg)
            face_id = int(image_name.split('.')[1])
            img_path = os.path.join(DATASET_PATH, image_name)

            # Загружаем и конвертируем изображение
            img = Image.open(img_path).convert('L')  # 'L' - grayscale
            img_np = np.array(img, 'uint8')

            faces.append(img_np)
            ids.append(face_id)
        except Exception as e:
            print(f"Ошибка при обработке {image_name}: {e}")

    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.write(MODEL_FILE)
        print(f"Модель обучена на {len(faces)} образцах!")
    else:
        print("Нет данных для обучения!")


# 7. Загрузка данных для обучения
def load_existing_dataset():
    """Загружает существующие данные из датасета для дообучения"""
    faces = []
    ids = []

    if not os.path.exists(DATASET_PATH):
        return faces, ids

    image_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.jpg')]

    for image_name in image_files:
        try:
            face_id = int(image_name.split('.')[1])
            img_path = os.path.join(DATASET_PATH, image_name)

            img = Image.open(img_path).convert('L')
            img_np = np.array(img, 'uint8')

            faces.append(img_np)
            ids.append(face_id)
        except:
            continue

    return faces, ids


# 8. Добавляет нового человека к существующей модели
def add_new_person(photo_folder: str):
    """Добавить нового человека к существующей модели"""
    global names

    name = photo_folder.split('_')[1]

    # Находим максимальный ID
    max_id = max(names.keys())
    new_id = max_id + 1

    # Добавляем в словарь
    names[new_id] = name

    # Добавляем фото
    samples = add_face_from_existing_photos(photo_folder, new_id)

    if samples > 0:
        # Дозаучиваем модель (LBPH поддерживает дообучение)
        faces, ids = load_existing_dataset()
        recognizer.update(faces, np.array(ids))
        recognizer.write(MODEL_FILE)

        # Сохраняем обновлённый словарь
        with open(NAMES_FILE, 'w', encoding='utf-8') as f:
            f.write(str(names))

        print(f"Добавлен новый человек: {name} (ID: {new_id})\n")
        main()


# 9. Загружается видео и распознаются лица на нём
def load_video(save_output=False):
    """Загружается видео и распознаются лица на нём"""
    video_file = input('Введите имя mp4-файла (без расширения) для обработки: ') + '.mp4'
    video = cv2.VideoCapture(video_file)
    video_writer = None
    if save_output:
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"recognized_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        print(f"Сохранение результата в: {output_file}")

    if not video.isOpened():
        print("Ошибка: не могу открыть видео файл!")
        return

    # Счётчик кадров для обработки не каждого кадра (оптимизация)
    frame_counter = 0
    # 1 - Обычный режим. Если 2, то обрабатываем каждый 3-й кадр. Оптимизация!
    if save_output:
        skip_frames = 1
    else:
        skip_frames = 2

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_counter += 1
        if not save_output:
            # Ниже идёт пропуск кадров. Для оптимизации, если не нужно сохранять - убрать!
            if frame_counter % skip_frames != 0:
                continue  # Пропускаем кадр

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Детекция лиц
        faces = face_cascade.detectMultiScale(gray_image, **DETECTION_PARAMS['video'])

        for (x, y, w, h) in faces:
            # Выделяем область лица
            roi_gray = gray_image[y:y + h, x:x + w]

            # Распознаём (только если модель обучена)
            if os.path.exists(MODEL_FILE):
                try:
                    id, confidence = recognizer.predict(roi_gray)

                    # Динамический порог в зависимости от освещения
                    threshold = 80  # Базовый порог

                    if confidence < threshold:
                        name = names.get(id, "Unknown")
                        color = (255, 250, 250)  # Белый

                        # Меняем цвет в зависимости от уверенности
                        if confidence > threshold * 0.7:  # 70% от порога
                            color = (255, 250, 250)  # Белый
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)  # Красный

                    # Форматируем текст
                    if name == "Unknown":
                        text = f"{name} ({confidence:.0f})"
                    else:
                        text = f"{name}"

                except Exception as e:
                    print(f"Ошибка предсказания: {e}")
                    continue  # Пропускаем это лицо
            else:
                name = "Детекция"
                color = (128, 128, 128)  # Серый - только детекция
                text = name

            thickness = 2
            if name != "Unknown":
                thickness = 3  # Более толстая рамка для распознанных

            # Рисуем прямоугольник
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            # Рисуем фон для текста с градиентом
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y - 35), (x + w, y), color, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Добавляем текст
            cv2.putText(frame, text, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 250, 250), 2)

        # Добавим красный полупрозрачный фон в духе зрения терминатора
        overlay = frame.copy()
        overlay[:] = (0, 0, 255)  # Красный
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Для большей серьёзности добавим нашему Т800 надпись "Идентификация"
        cv2.putText(overlay, 'IDENTIFICATION', (50, 100), font, 2, (255, 250, 250), 3, cv2.LINE_AA)
        alpha = 0.7
        beta = 0.3
        gamma = 0  # Смещение яркости
        result = cv2.addWeighted(frame, alpha, overlay, beta, gamma)  # Кадр на выходе

        if video_writer is not None:
            video_writer.write(result)

        # Показываем кадр
        cv2.imshow('Распознавание лиц', result)

        # Выход по 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    if video_writer is not None:
        video_writer.release()
        print(f"\nВидео успешно сохранено!")
    cv2.destroyAllWindows()


# 10. Добавим фото для обучения
def add_photo_for_learning():
    """Добавим фото для обучения"""
    global names

    print("Модель не обучена. Сначала нужно добавить лица.")
    print("1 - Добавить из готовых фото")
    print("2 - Снять с камеры (если нужно)")
    print("3 - Только детекция (без распознавания)")

    response = input("Выберите вариант (1/2/3): ").strip()
    print('')

    if response == '1':
        # Используем готовые фото из папки 'photos_****'
        # Получаем список всех объектов (файлов и папок) в текущей директории
        folders_with_word_current = []
        for item in os.listdir('.'):
            # Проверяем, является ли объект папкой и содержит ли имя нужное слово
            if os.path.isdir(item) and TARGET_WORD in item:
                folders_with_word_current.append(item)
        names[0] = 'Unknown'  # Добавим нулевой ID для ноунеймов
        for i, item in enumerate(folders_with_word_current):
            names[i + 1] = item.split('_')[1]  # разбить имя папки на "photos" и имя
        with open(NAMES_FILE, 'w', encoding='utf-8') as file:
            names_string = str(names)
            file.write(names_string)
        # У нас есть словарь с ID и именами, а также папка
        for index, item in enumerate(folders_with_word_current):
            add_face_from_existing_photos(item, index + 1)

        total_samples = 0
        for index, item in enumerate(folders_with_word_current):
            samples = add_face_from_existing_photos(item, index + 1)
            total_samples += samples
        if total_samples > 0:
            train_recognizer()  # Обучаем один раз на всех данных

    elif response == '2':
        # Старый способ с камерой (закомментирован, но можно использовать)
        print("Этот вариант временно недоступен. Используйте вариант 1.")
        add_face_to_dataset()  # Раскомментируйте, если нужно
        # add_face_from_existing_photos("photos_dima")  # Используем фото как запасной вариант

    elif response == '3':
        print("Работаю только в режиме детекции (без распознавания)")

    else:
        print("Команда не распознана")
        add_photo_for_learning()

    # Заполняем словарь из файла
    with open(NAMES_FILE, 'r', encoding='utf-8') as f:
        # Читаем содержимое файла и превращаем строку в словарь
        names = {}
        names = ast.literal_eval(f.read())

    load_video()


# 11. Основная функция
def main():
    """Основная функция. Точка входа."""
    global names

    # Проверяем, обучена ли модель
    if os.path.exists(MODEL_FILE) and os.path.exists(NAMES_FILE):
        print("Найдена обученная модель. Выберите вариант.")
        print("1 - Просмотреть видео")
        print("2 - Сохранить видео")
        print("3 - Дообучить модель")
        print("4 - Обучить модель с нуля")
        use_existing = input("Выберите вариант (1/2/3/4): ")
        print('')
        if use_existing.lower() == '1':
            recognizer.read(MODEL_FILE)
            with open(NAMES_FILE, 'r', encoding='utf-8') as f:
                names = ast.literal_eval(f.read())
            print(f"Загружена модель с {len(names) - 1} людьми")
            load_video()
        if use_existing.lower() == '2':
            recognizer.read(MODEL_FILE)
            with open(NAMES_FILE, 'r', encoding='utf-8') as f:
                names = ast.literal_eval(f.read())
            print(f"Загружена модель с {len(names) - 1} людьми")
            save_output = True
            load_video(save_output)
        elif use_existing.lower() == '3':
            add_new_person(input('Введите имя папки: '))
        elif use_existing.lower() == '4':
            add_photo_for_learning()
        else:
            print("Команда не распознана")
            main()


# Запуск программы
if __name__ == "__main__":
    main()
