"""Program for detecting and recognizing faces in video (Terminator-Style)"""

import cv2
import numpy as np
import os
import ast
from PIL import Image
from faces_video_config import *
from datetime import datetime

print('\nHi! I am a facial recognition program.\n') # Hellow


# 1. Initializing the face detector (using a Haar cascade)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# 2. Building a Face Recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer.create()


# 3. Dictionary for ID and name mapping. Populated from a file.
with open(NAMES_FILE, 'r', encoding='utf-8') as f:
    # Read the contents of the file and convert the string into a dictionary
    names = ast.literal_eval(f.read())


# 4. Path to the dataset
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)


# 5.1 A function for adding a face to the database using a photo from a camera
def add_face_to_dataset():
    """Adding my face to the database using a webcam photo"""
    face_id = 1  # ID for your faces
    count = 0

    cap = cv2.VideoCapture(0)  # Web camera
    print("Creating a dataset. Look into the camera...")
    print("I'm collecting 30 samples of your face...")

    while count < 30:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecting faces
        faces = face_cascade.detectMultiScale(gray, gray, **DETECTION_PARAMS['photo'])

        for (x, y, w, h) in faces:
            # Draw a rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # We only save face
            face_roi = gray[y:y + h, x:x + w]

            # Enlarging the face to standard size
            face_resized = cv2.resize(face_roi, (200, 200))

            # Save to dataset
            count += 1
            cv2.imwrite(f"{DATASET_PATH}/User.{face_id}.{count}.jpg", face_resized)
            print(f"Sample saved {count}/30")

        # Show the process
        cv2.imshow('Добавление лица в базу', frame)

        # Delay to collect different angles
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("The dataset has been created!")

    # Training the model
    train_recognizer()


# 5.2 A function for adding faces to a database from existing photos
def add_face_from_existing_photos(photo_folder: str, photo_id: int):
    """
    Loads ready-made photos from a folder instead of taking them
    from the camera photo folder - a folder with photos (jpg/png)
    """
    face_id = photo_id  # ID лица

    if not os.path.exists(photo_folder):
        print(f"Error: folder '{photo_folder}' not found!")
        print(f"Create folder '{photo_folder}' and add photos there")
        return

    # Getting a list of photos
    photo_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    photo_files = [f for f in os.listdir(photo_folder)
                   if os.path.splitext(f)[1].lower() in photo_extensions]

    if not photo_files:
        print(f"In folder '{photo_folder}' no images!")
        return

    print(f"Found {len(photo_files)} photos. Processing...")

    count = 0
    for i, photo_file in enumerate(photo_files):
        photo_path = os.path.join(photo_folder, photo_file)

        try:
            # Load photos
            img = cv2.imread(photo_path)
            if img is None:
                print(f"Can't load: {photo_file}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detecting faces in photos
            faces = face_cascade.detectMultiScale(gray, **DETECTION_PARAMS['photo'])

            if len(faces) == 0:
                print(f"No faces in the photo {photo_file}")
                continue

            # Take the first face found
            for (x, y, w, h) in faces[:1]:  # Берём только первое лицо
                # Only save face
                face_roi = gray[y:y + h, x:x + w]

                # Enlarging the face to standard size
                face_resized = cv2.resize(face_roi, (200, 200))

                # Save to dataset
                count += 1
                cv2.imwrite(f"{DATASET_PATH}/User.{face_id}.{count}.jpg", face_resized)
                print(f"Photo {i + 1}/{len(photo_files)}: face saved {count}")

                # Show the found face (optional)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow(f"Face found in {photo_file}", img)
                cv2.waitKey(500)  # Showing 0.5 seconds
                cv2.destroyWindow(f"Face found in {photo_file}")

        except Exception as e:
            print(f"Error while processing {photo_file}: {e}")
            continue

    cv2.destroyAllWindows()
    print(f"Dataset created! {count} face samples saved.")

    return count  # Return the number of samples added


# 6. Model training function
def train_recognizer():
    """Model training function"""
    faces = []
    ids = []

    # Checking if there are files in the dataset
    image_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.jpg')]

    if not image_files:
        print("There are no images in the dataset!")
        return

    for image_name in image_files:
        try:
            # Extracting ID from file name (format: User.id.number.jpg)
            face_id = int(image_name.split('.')[1])
            img_path = os.path.join(DATASET_PATH, image_name)

            # Extracting ID from file name
            img = Image.open(img_path).convert('L')  # 'L' - grayscale
            img_np = np.array(img, 'uint8')

            faces.append(img_np)
            ids.append(face_id)
        except Exception as e:
            print(f"Error while processing {image_name}: {e}")

    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.write(MODEL_FILE)
        print(f"The model is trained on {len(faces)} samples.!")
    else:
        print("No data for training!")


# 7. Loading training data
def load_existing_dataset():
    """Loads existing data from the dataset for additional training"""
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


# 8. Adding a new person to an existing model
def add_new_person(photo_folder: str):
    """Add a new person to an existing model"""
    global names

    name = photo_folder.split('_')[1]

    # Finding the maximum ID
    max_id = max(names.keys())
    new_id = max_id + 1

    # Add to dictionary
    names[new_id] = name

    # Adding a photo
    samples = add_face_from_existing_photos(photo_folder, new_id)

    if samples > 0:
        # Retraining the model (LBPH supports retraining)
        faces, ids = load_existing_dataset()
        recognizer.update(faces, np.array(ids))
        recognizer.write(MODEL_FILE)

        # Saving the updated dictionary
        with open(NAMES_FILE, 'w', encoding='utf-8') as f:
            f.write(str(names))

        print(f"New person added: {name} (ID: {new_id})\n")
        main()


# 9. The video is loaded and faces are recognized in it.
def load_video(save_output=False):
    """The video is loaded and faces are recognized in it."""
    video_file = input('Enter the name of the mp4 file (without extension) to process: ') + '.mp4'
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
        print(f"Saving the result in: {output_file}")

    if not video.isOpened():
        print("Error: I can't open the video file!")
        return

    # Frame counter for processing not every frame (optimization)
    frame_counter = 0
    # 1 - Normal mode. If 2, then we process every 3rd frame. Optimization!
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
            # Below is the frame skipping. For optimization purposes, if you don't need to save it, remove it!
            if frame_counter % skip_frames != 0:
                continue  # Skipping a frame

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(gray_image, **DETECTION_PARAMS['video'])

        for (x, y, w, h) in faces:
            # Select the face area
            roi_gray = gray_image[y:y + h, x:x + w]

            # Recognize (only if the model is trained)
            if os.path.exists(MODEL_FILE):
                try:
                    id, confidence = recognizer.predict(roi_gray)

                    # Dynamic threshold depending on lighting
                    threshold = 80  # Basic threshold

                    if confidence < threshold:
                        name = names.get(id, "Unknown")
                        color = (255, 250, 250)  # White

                        # Change color depending on confidence
                        if confidence > threshold * 0.7:  # 70% of the threshold
                            color = (255, 250, 250)  # White
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)  # Red

                    # Formatting the text
                    if name == "Unknown":
                        text = f"{name} ({confidence:.0f})"
                    else:
                        text = f"{name}"

                except Exception as e:
                    print(f"Prediction error: {e}")
                    continue  # Let's skip this face
            else:
                name = "Detection"
                color = (128, 128, 128)  # Gray - detection only
                text = name

            thickness = 2
            if name != "Unknown":
                thickness = 3  # Thicker frame for recognized

            # Draw a rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            # Drawing a gradient background for text
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y - 35), (x + w, y), color, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Adding text
            cv2.putText(frame, text, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 250, 250), 2)

        # Let's add a red translucent background in the spirit of the Terminator's vision
        overlay = frame.copy()
        overlay[:] = (0, 0, 255)  # Red
        font = cv2.FONT_HERSHEY_SIMPLEX
        # To make it look more serious, let's add the inscription "Identification" to our T800.
        cv2.putText(overlay, 'IDENTIFICATION', (50, 100), font, 2, (255, 250, 250), 3, cv2.LINE_AA)
        alpha = 0.7     # Transparency of the original
        beta = 0.3      # Overlay transparency
        gamma = 0       # Brightness shift
        result = cv2.addWeighted(frame, alpha, overlay, beta, gamma)  # Frame at the exit

        if video_writer is not None:
            video_writer.write(result)

        # Showing a frame
        cv2.imshow('Facial recognition', result)

        # Exit by 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved successfully!")
    cv2.destroyAllWindows()


# 10. Let's add photos for training
def add_photo_for_learning():
    """Let's add photos for training"""
    global names

    print("The model is untrained. We need to add faces first.")
    print("1 - Add from ready-made photos")
    print("2 - Remove from camera (if necessary)")
    print("3 - Detection only (no recognition)")

    response = input("Select an option (1/2/3): ").strip()
    print('')

    if response == '1':
        # We use ready-made photos from the folder 'photos_****'
        # We get a list of all objects (files and folders) in the current directory
        folders_with_word_current = []
        for item in os.listdir('.'):
            # We check whether the object is a folder and whether the name contains the required word
            if os.path.isdir(item) and TARGET_WORD in item:
                folders_with_word_current.append(item)
        names[0] = 'Unknown'  # Let's add a zero ID for no-names
        for i, item in enumerate(folders_with_word_current):
            names[i + 1] = item.split('_')[1]  # split the folder name into "photos" and the name
        with open(NAMES_FILE, 'w', encoding='utf-8') as file:
            names_string = str(names)
            file.write(names_string)
        # We have a dictionary with IDs and names, and a folder
        for index, item in enumerate(folders_with_word_current):
            add_face_from_existing_photos(item, index + 1)

        total_samples = 0
        for index, item in enumerate(folders_with_word_current):
            samples = add_face_from_existing_photos(item, index + 1)
            total_samples += samples
        if total_samples > 0:
            train_recognizer()  # We train once on all data

    elif response == '2':
        # Old method with camera (commented, but can be used)
        print("This option is temporarily unavailable. Please use option 1.")
        add_face_to_dataset()  # Uncomment if necessary
        # add_face_from_existing_photos("photos_dima")  # We use photos as a backup option

    elif response == '3':
        print("I work only in detection mode (without recognition)")

    else:
        print("Command not recognized")
        add_photo_for_learning()

    # Filling the dictionary from a file
    with open(NAMES_FILE, 'r', encoding='utf-8') as f:
        # Read the contents of the file and convert the string into a dictionary
        names = {}
        names = ast.literal_eval(f.read())

    load_video()


# 11. Main function
def main():
    """Main function. Entry point."""
    global names

    # Checking if the model is trained
    if os.path.exists(MODEL_FILE) and os.path.exists(NAMES_FILE):
        print("A trained model was found. Please select an option.")
        print("1 - Watch the video")
        print("2 - Save video")
        print("3 - Retrain the model")
        print("4 - Train a model from scratch")
        use_existing = input("Select an option (1/2/3/4): ")
        print('')
        if use_existing.lower() == '1':
            recognizer.read(MODEL_FILE)
            with open(NAMES_FILE, 'r', encoding='utf-8') as f:
                names = ast.literal_eval(f.read())
            print(f"Model loaded with {len(names) - 1} people")
            load_video()
        if use_existing.lower() == '2':
            recognizer.read(MODEL_FILE)
            with open(NAMES_FILE, 'r', encoding='utf-8') as f:
                names = ast.literal_eval(f.read())
            print(f"Model loaded with {len(names) - 1} people")
            save_output = True
            load_video(save_output)
        elif use_existing.lower() == '3':
            add_new_person(input('Enter the folder name: '))
        elif use_existing.lower() == '4':
            add_photo_for_learning()
        else:
            print("Command not recognized")
            main()


# Launching the program
if __name__ == "__main__":
    main()
