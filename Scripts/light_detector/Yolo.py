import torch
import cv2
import numpy as np
import glob
from tqdm import tqdm
import os
import time
import Scripts.light_detector.Median_cut as MC

def start(input_path='./Scripts/light_detector/to_predict',view_path = './Scripts/light_detector/prediction_view'):
    start_time = time.time()
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', './models/best.pt' ,force_reload=False, trust_repo=True )
    if input_path == '':
        print('Wrong input')
        return
    # Declaring some variables    
    TABLE_CONFIDENCE = 0.1
    CELL_CONFIDENCE = 0.1
    CONFIDENCE = 0.1

    # Bounding Boxes color scheme
    ALPHA = 0.2
    TABLE_BORDER = (0, 0, 255)
    CELL_BORDER = (255, 0, 0)

    os.makedirs(view_path, exist_ok=True)
    
    # Run the Inference and draw predicted bboxes
    dir = []
    dir.extend(glob.glob(input_path+'/*.jpg'))
    #print(dir)
    print("Yolo model start...\n")
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
            image = cv2.putText(image, "point", (table_bbox[0],  table_bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,TABLE_BORDER , 1)


        for cell_bbox in cell_bboxes:
            cv2.rectangle(image, (cell_bbox[0], cell_bbox[1]),
                        (cell_bbox[2], cell_bbox[3]), CELL_BORDER, 1)
            image = cv2.putText(image, "Area", (cell_bbox[0], cell_bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,CELL_BORDER , 1)

        image_new = cv2.addWeighted(overlay, ALPHA, image, 1-ALPHA, 0)
        file_name = os.path.basename(i).split('.')[0]
        cv2.imwrite(f'{view_path}/{file_name}.jpg', image_new)
        print(arra)
        print(f"delete index {index}")
        arra = np.delete(arra,index,axis=0)
        print(arra)
        MC.medium_cut(f'./Scripts/LDR2HDR/images_HDR/{file_name}.exr', arra.tolist(),f'./Scripts/LDR2HDR/images_LDR/{file_name}.jpg',depth=True)
    print(f"Medium cut time cost{time.time()-start_time}")
    
if __name__ == '__main__':
    input_path = './Scripts/light_detector/to_predict'
    output_path = './Scripts/light_detector/prediction_view'
    start(input_path,output_path)