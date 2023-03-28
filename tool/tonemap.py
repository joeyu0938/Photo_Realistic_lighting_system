import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path
path = './Loader/to_predict'
output_path = './Loader/to_predict_tonemap'
def tonemap(path,output_path):
    dir = []
    dir.extend(glob.glob(f'{path}/*.exr'))
    for i in tqdm(dir):
        file_name = Path(i).stem # 取得除去副檔名的檔名
        im = cv2.imread(i, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        tonemapDurand = cv2.createTonemap(gamma = 2.2)
        ldrDurand = tonemapDurand.process(im)
        # output
        im2_8bit = np.clip(ldrDurand * 255, 0, 255).astype('uint8')        
        cv2.imwrite(f'{output_path}/{file_name}.jpg', im2_8bit)
         
if __name__ == '__main__':
    tonemap(path,output_path)