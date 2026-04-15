"""Configuration for face recognition in video"""

DETECTION_PARAMS = {
    'photo': {'scaleFactor': 1.2, 'minNeighbors': 8, 'minSize': (30, 30)},
    'video': {'scaleFactor': 1.1, 'minNeighbors': 9, 'minSize': (50, 50)}
}                           # recognition accuracy parameters
TARGET_WORD = "photos_"     # template for naming photo folders, after "_" comes the name (English!)
DATASET_PATH = "dataset"    # a folder containing a database of faces that the program extracted from photographs
NAMES_FILE = 'names.txt'    # a file with the IDs and names of people the program will recognize
MODEL_FILE = 'trainer.yml'  # the resulting data model for face recognition
