import glob
import os
import tonemap
from tqdm import tqdm
import shutil

# path = 'C:/Users/User/Desktop/yolov7-main/data/lights/images/Tend_hdr_train'
# output_path = 'C:/Users/User/Desktop/LTL_learning/Loader/to_predict_tonemap'
# tonemap.tonemap(path,output_path)

send_path = 'C:/Users/User/Desktop/yolov7-main/data/lights/images/tmp/*.exr'
for number, filename in tqdm(enumerate(glob.glob(f'Loader/to_predict_tonemap_big/*.jpg'))):
        try:
            shutil.copy2(filename, f"{send_path}/{os.path.basename(filename).split('.')[0].split('_')[1]}/{os.path.basename(filename).split('.')[0].split('_')[0]}.exr")
            #os.remove(filename)
            # os.rename(filename, f"{send_path}/{os.path.basename(filename).split('.')[0].split('_')[1]}/\
            #     {os.path.basename(filename).split('.')[0].split('_')[0]}.exr")
        except OSError as e:
            print("Something happened:", e)