import os
import shutil
import random

src = r"C:\Users\Lenovo\OneDrive\Desktop\deepfake_detection_project\merged_dataset"
dest = r"C:\Users\Lenovo\OneDrive\Desktop\deepfake_detection_project\dataset_split"

split_ratio = (0.7, 0.15, 0.15)

for label in ["real", "fake"]:

    files = os.listdir(os.path.join(src, label))
    random.shuffle(files)

    train_end = int(len(files)*split_ratio[0])
    val_end = train_end + int(len(files)*split_ratio[1])

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for folder, file_list in zip(
        ["train", "validation", "test"],
        [train_files, val_files, test_files]
    ):

        os.makedirs(os.path.join(dest, folder, label), exist_ok=True)

        for file in file_list:
            src_path = os.path.join(src, label, file)
            dst_path = os.path.join(dest, folder, label, file)

            shutil.copy(src_path, dst_path)

print("Dataset split completed")