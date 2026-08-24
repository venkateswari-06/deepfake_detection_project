from PIL import Image
import os

dataset_path = r"C:\Users\Lenovo\OneDrive\Desktop\deepfake_detection_project\dataset_split"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        path = os.path.join(root, file)
        try:
            img = Image.open(path)
            img.verify()
        except:
            print("Removing corrupted file:", path)
            os.remove(path)

print("Dataset cleaned")