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
    CONFIDENCE = 0.0
    LOW_PLACE =  140 # 把低於一個高度的地板光取消
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
        low = b[1].tolist()[0]
        index = []
        for idx,v in enumerate(confidence):
            if v <CONFIDENCE:
                index.append(idx)
                print("delete low confidence")
        for idx,v in enumerate(low):
            if v > LOW_PLACE: 
                index.append(idx)
                print("delete low place light")    
        #跟yolo的xy 倒過來了
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
            # 這裡row 的 屬性是yolo 的
            if row['class'] == 0 and row['confidence'] > TABLE_CONFIDENCE and row['ymin']< LOW_PLACE:
                table_bboxes.append([int(row['xmin']), int(row['ymin']),
                                    int(row['xmax']), int(row['ymax']),row['confidence']])
            if row['class'] == 1 and row['confidence'] > CELL_CONFIDENCE and row['ymin']< LOW_PLACE:
                cell_bboxes.append([int(row['xmin']), int(row['ymin']),
                                    int(row['xmax']), int(row['ymax']),row['confidence']])

        image = cv2.imread(i)
        image = cv2.resize(image,(1536, 768), interpolation=cv2.INTER_AREA)
        overlay = image.copy()
        for table_bbox in table_bboxes:
            cv2.rectangle(image, (table_bbox[0]*3, table_bbox[1]*3),
                        (table_bbox[2]*3, table_bbox[3]*3), TABLE_BORDER, 1)
            image = cv2.putText(image, f"point:{table_bbox[4]}", (table_bbox[0]*3,  table_bbox[1]*3 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,TABLE_BORDER , 2)


        for cell_bbox in cell_bboxes:
            cv2.rectangle(image, (cell_bbox[0]*3, cell_bbox[1]*3),
                        (cell_bbox[2]*3, cell_bbox[3]*3), CELL_BORDER, 1)
            image = cv2.putText(image, f'area:{cell_bbox[4]}', (cell_bbox[0]*3, cell_bbox[1]*3 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,CELL_BORDER , 2)

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