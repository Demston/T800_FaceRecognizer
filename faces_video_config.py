"""Конфиг для распознавания лиц на видео"""

DETECTION_PARAMS = {
    'photo': {'scaleFactor': 1.2, 'minNeighbors': 8, 'minSize': (30, 30)},
    'video': {'scaleFactor': 1.1, 'minNeighbors': 9, 'minSize': (50, 50)}
}                           # параметры точности распознавания
TARGET_WORD = "photos_"     # шаблон для имени папок с фото (после "_" идёт имя на англ.)
DATASET_PATH = "dataset"    # папка с базой лиц, которые программа вытащила с фотографий
NAMES_FILE = 'names.txt'    # файл с id и именами людей, которых программа будет распознавать
MODEL_FILE = 'trainer.yml'  # полученная модель данных для распознавания лиц
