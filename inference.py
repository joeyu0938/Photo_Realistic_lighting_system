import os
import LDR2HDRinference as L2H
import sys
import torch
import cv2
#sys.path.append("../utils")
import tool.Filter_pytorch as FP
import tool.Multi_proccess_tend as MP
import tool.tonemap as TM
import Median_cut as MC
import glob
from tqdm import tqdm
from pathlib import Path
import numpy as np

def start():
    args_FP,remain= FP.set_parser().parse_known_args()
    args_L2H,unknown = L2H.parse_arguments(remain)
    #Setting file type 
    if(args_FP.source_type == 'HDR'):
        args_FP.extension = ['.exr','.hdr']
    elif(args_FP.source_type == 'LDR'):
        args_FP.extension = ['.jpg','png']
    print(args_FP)
    print(args_L2H)
    L2H.main(args_L2H)
    MP.start(args_FP)
    # ---------------------------
    # 下面是跑pre trained model 的 code
    # Load fine-tuned custom model
  
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', './best.pt' ,force_reload=False, trust_repo=True )
     
    # Declaring some variables    
    TABLE_CONFIDENCE = 0.2
    CELL_CONFIDENCE = 0.2
    CONFIDENCE = 0.2
    OUTPUT_DIR = './output'

    # Bounding Boxes color scheme
    ALPHA = 0.2
    TABLE_BORDER = (0, 0, 255)
    CELL_FILL = (0, 0, 200)
    CELL_BORDER = (0, 0, 255)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    
    
    # 將 to_predict 的結果先進行 tonemapping 之後再批量放進 model 中產生 label
    img_path = './Loader/to_predict'
    ldr_path = './Loader/images_LDR'
    output_path = './Loader/to_predict_tonemap'
    TM.tonemap(img_path,output_path)
    
    
    # Run the Inference and draw predicted bboxes
    dir = []
    dir.extend(glob.glob(f'{output_path}/*.jpg'))
    #print(dir)
    for i in tqdm(dir):
        results = model(i)
        df = results.pandas().xyxy[0]
        #print(df)
        res = []
        
        for column in df.columns:
            if  column != 'name':     
                li = df[column].tolist()
                res.append(li)    
        
        res=np.asarray(res)
        #print(res)
        b = np.vsplit(res, 6)
        confidence = b[4].tolist()[0]
        index = []
        for idx,v in enumerate(confidence):
            if v <CONFIDENCE:
                index.append(idx)
            else:
                print("delete low confidence")
        xmin = b[1]
        ymin = b[0]
        xmax = b[3]
        ymax = b[2]
        Class = b[5]
        x = xmin
        y = ymin
        width = np.subtract(ymax,ymin)
        length = np.subtract(xmax,xmin)
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        width = width.astype(np.int32)
        length = length.astype(np.int32)
        Class = Class.astype(np.int32)
        arra = np.concatenate((Class,y,x,width,length),axis=0) 
        arra = arra.T
            
        table_bboxes = []
        cell_bboxes = []
        tempt = []
        for _, row in df.iterrows():
            if row['class'] == 0 and row['confidence'] > TABLE_CONFIDENCE:
                table_bboxes.append([int(row['xmin']), int(row['ymin']),
                                    int(row['xmax']), int(row['ymax'])])
            if row['class'] == 1 and row['confidence'] > CELL_CONFIDENCE:
                cell_bboxes.append([int(row['xmin']), int(row['ymin']),
                                    int(row['xmax']), int(row['ymax'])])

        image = cv2.imread(i)
        overlay = image.copy()
        for table_bbox in table_bboxes:
            cv2.rectangle(image, (table_bbox[0], table_bbox[1]),
                        (table_bbox[2], table_bbox[3]), TABLE_BORDER, 1)

        for cell_bbox in cell_bboxes:
            cv2.rectangle(overlay, (cell_bbox[0], cell_bbox[1]),
                        (cell_bbox[2], cell_bbox[3]), CELL_FILL, -1)
            cv2.rectangle(image, (cell_bbox[0], cell_bbox[1]),
                        (cell_bbox[2], cell_bbox[3]), CELL_BORDER, 1)

        image_new = cv2.addWeighted(overlay, ALPHA, image, 1-ALPHA, 0)
        image = output_path.split('/')[-1]
        image_filename = str(Path(i).stem)
        cv2.imwrite(f'{OUTPUT_DIR}/{image}.jpg', image_new)
        print(arra)
        print(f"delete index {index}")
        arra = np.delete(arra,index,axis=0)
        print(arra)
        MC.medium_cut(f'{img_path}/{image_filename}.exr', arra.tolist(),f'{ldr_path}/upload.jpg')
        
if __name__ == '__main__':
    start()