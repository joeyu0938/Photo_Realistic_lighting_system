# may be delete
import cv2
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path
path = 'C:/Users/User/Desktop/Trainig Dataset/Image'
output_path = 'C:/Users/User/Desktop/yolov7-main_test/data/lights/images/train'
dir = []
dir.extend(glob.glob(f'{path}/*.jpg'))
print(f'find: \n{dir}\n')
for i in tqdm(dir):
    file_name = Path(i).stem
    input_img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
    print(input_img.shape)
    input_img = cv2.resize(input_img, (4096,2048), interpolation=cv2.INTER_LANCZOS4)
    input_img = cv2.resize(input_img, (2048,1024), interpolation=cv2.INTER_CUBIC)
    input_img = cv2.resize(input_img, (1024,512), interpolation=cv2.INTER_AREA)
    input_img = cv2.resize(input_img, (512,256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f'{output_path}/{file_name}.jpg', input_img)  