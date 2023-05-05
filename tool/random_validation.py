import glob
import os
import random
from tqdm import tqdm
import shutil
label_file_path = 'C:/Users/User/Desktop/yolov7-main/data/lights/labels'
image_file_path = 'C:/Users/User/Desktop/yolov7-main/data/lights/images'
label_dir = glob.glob(label_file_path+'/whole/*txt')
image_dir = glob.glob(image_file_path+'/whole/*jpg')
print(f"Label count {len(label_dir)} , image count {len(image_dir)}")
k  = 100 # validation count
x = random.sample(label_dir,k)
y = set(label_dir) - set(x)
print(f"pick {x}")
print(f"rest {y}")
if input("Continue: ")!= 'N':
    for i in tqdm(x):
        name = os.path.basename(i).split('.')[0]
        shutil.copy2(i, f"{label_file_path}/val/{name}.txt")
        shutil.copy2(image_file_path+f'/whole/{name}.jpg', f"{image_file_path}/val/{name}.jpg")
    for i in tqdm(y):
        name = os.path.basename(i).split('.')[0]
        shutil.copy2(i, f"{label_file_path}/train/{name}.txt")
        shutil.copy2(image_file_path+f'/whole/{name}.jpg', f"{image_file_path}/train/{name}.jpg")
