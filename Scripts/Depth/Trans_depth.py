import torch
import cv2
import numpy as np
import glob
from tqdm import tqdm
import os
import time

def start(input_path='./Scripts/LDR2HDR/images_LDR',output_path='./Scripts/Depth/image_depth'):
    start_time = time.time()
    if input_path == '' or output_path=='':
        print('Wrong input')
        return
    input_files = glob.glob(input_path+'/*.jpg')
    # ---------------------------
    # 下面是跑depth pretrained model 的 code
  
    #Depth part
    #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    print("Predicting depth....\n")
    print(input_files)
    dir = list()
    dir.extend(input_files)
    for i in tqdm(input_files):
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)
        # 深度會抓不準因為下面的空白資料 必須要完整
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(256,512),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        #normalize
        output = prediction.cpu().numpy()
        output[output<1] = 1
        output = np.max(output)/output
        file_name = os.path.basename(i).split('.')[0]
        np.save(f'{output_path}/{file_name}.npy', output)
        #cv2.imshow('depth',output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(f'{output_path}/{file_name}_depth.jpg',output*8)
    print(f"Depth estimation Time cost {time.time() - start_time}\n")
if __name__ == '__main__':
    input_path = './Scripts/LDR2HDR/images_LDR'
    output_path = './Scripts/Depth/image_depth'
    start(input_path,output_path)