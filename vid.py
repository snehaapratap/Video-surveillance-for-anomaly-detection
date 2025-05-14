import os
import cv2
import numpy as np
from tqdm import tqdm

# === CONFIG ===
train_path = "C:\\Users\\jaiga\\DeepEYE\\train_data"
frame_save_path = os.path.join(train_path, 'frames')
fps_skip = 5
output_file = "training.npy"

if not os.path.exists(frame_save_path):
    os.makedirs(frame_save_path)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

# === PROCESS VIDEOS ===
all_frames = []
train_videos = os.listdir(train_path)
for video_file in tqdm(train_videos, desc="Processing Videos"):
    if not video_file.endswith(('.mp4', '.avi', '.mov')):
        continue
    video_path = os.path.join(train_path, video_file)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % fps_skip == 0:
            gray_frame = preprocess_frame(frame)
            all_frames.append(gray_frame)
            frame_name = os.path.join(frame_save_path, f"{video_file}_{saved_frame_id:03d}.jpg")
            cv2.imwrite(frame_name, gray_frame)
            saved_frame_id += 1

        frame_id += 1

    cap.release()

# === Convert and Normalize ===
data = np.array(all_frames)  # shape: (N, 227, 227)
data = np.transpose(data, (1, 2, 0))  # (227, 227, N)

# Normalize and clip
data = (data - np.mean(data)) / np.std(data)
data = np.clip(data, 0, 1)

# Save
np.save(output_file, data)
print(f"Saved preprocessed training data to {output_file}")
