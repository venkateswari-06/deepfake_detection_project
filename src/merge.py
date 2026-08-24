import os
import shutil

src1 = r"C:\Users\Lenovo\OneDrive\Desktop\deepfake_detection_project\DFDC_Dataset"
src2 = r"C:\Users\Lenovo\OneDrive\Desktop\deepfake_detection_project\adham7elym"
dest = r"C:\Users\Lenovo\OneDrive\Desktop\deepfake_detection_project\merged_dataset"

for label in ["real", "fake"]:
    os.makedirs(os.path.join(dest, label), exist_ok=True)

    for src in [src1, src2]:
        path = os.path.join(src, label)

        if os.path.exists(path):
            for file in os.listdir(path):
                src_file = os.path.join(path, file)

                new_name = os.path.basename(src) + "_" + file
                dst_file = os.path.join(dest, label, new_name)

                shutil.copy(src_file, dst_file)

print("Datasets merged successfully!")