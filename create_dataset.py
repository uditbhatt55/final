import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):

    folder_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(folder_path):
        continue

    print("Processing:", dir_)

    # ⚡ limit images for speed
    for img_path in os.listdir(folder_path)[:300]:

        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(folder_path, img_path))

        if img is None:
            continue

        # ⚡ resize for speed
        img = cv2.resize(img, (256, 256))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# save
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ Dataset Created Successfully")