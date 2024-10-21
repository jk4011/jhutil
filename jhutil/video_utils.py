import cv2
import os
import random


def video_to_images(video_path, output_folder, total_frames=-1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break
        frame_count += 1
    
    if total_frames == -1:
        total_frames = frame_count
    selected_frames = random.sample(range(frame_count), total_frames)
    
    video = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        
        if frame_count in selected_frames:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
        
        frame_count += 1

    print(f"Video converted. {total_frames} frames saved to {output_folder}.")
