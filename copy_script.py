import os
import shutil

img_dir = './FRLL/images/bonafide/raw'

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    new_path = img_path.replace("/images/", "/features_scale_1/").replace(".jpg", ".pt")
    dest_dir = './FRLL/features_scale_1/bonafide/raw'
    shutil.copy(img_path, new_path)
