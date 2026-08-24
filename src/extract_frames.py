import cv2
import os

# Input dataset folders
real_path = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\deepfake_detection_project\\DFDC_Dataset\\Real"
fake_path = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\deepfake_detection_project\\DFDC_Dataset\\Fake"

# Output frame folders
real_output = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\deepfake_detection_project\\extracted_frames\\real"
fake_output = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\deepfake_detection_project\\extracted_frames\\fake"
os.makedirs(real_output, exist_ok=True)
os.makedirs(fake_output, exist_ok=True)

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_name, frame)

        frame_count += 1

    cap.release()


# Process real videos
for video in os.listdir(real_path):
    video_file = os.path.join(real_path, video)
    extract_frames(video_file, real_output)

# Process fake videos
for video in os.listdir(fake_path):
    video_file = os.path.join(fake_path, video)
    extract_frames(video_file, fake_output)

print("Frame extraction completed!")