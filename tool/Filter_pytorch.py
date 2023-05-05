#獲得趨勢圖
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import torch.nn as nn
import numpy as np
from torch.optim import optimizer
from sklearn import datasets
#import matplotlib.pyplot as plt
import argparse
import cv2
import time
import random
import glob
from pathlib import Path
from tqdm import tqdm
#import torch.multiprocessing as mp

#filter 裡面的神經網路
class Filter(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(Filter, self).__init__()
        # define layers
        self.linear = nn.Linear(input_dim, output_dim,bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)
    
def set_parser():
    #Setting parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--source', type=str, default='*/images_HDR', help='Choose source folder')
    parser.add_argument('-st','--source_type', type=str, default='HDR', help='Image type of source')
    parser.add_argument('-to','--tend_output',type= str, default='Loader/to_predict',help='Tend and Hdr mix')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--Learning_rate', type=float,default=0.05, help='learning rate for 3d regression')
    parser.add_argument('--epoch', type=int,default=4, help='epoch for training 3d regression')
    parser.add_argument('--plot', type=bool,default=False, help='plot for 3d regression')
    parser.add_argument('--filter_size', type=int, default=4, help='filter_size for 3d regression')
    return parser

#建立輸入參數
def create_parser():
    opt = set_parser()
    opt = opt.parse_args() 
    
    #Setting file type 
    if(opt.source_type == 'HDR'):
        opt.extension = ['.exr','.hdr']
    elif(opt.source_type == 'LDR'):
        opt.extension = ['.jpg','png']
    
    return opt

# 輸入 array ,目標座標 x ,y , 設定參數 , 原圖  # 意義 取得filter  linear regression 的 參數 y = ax+by+bias  # 輸出array
def train(arr,tar_x,tar_y,parser,input):
    

    #create xy feature
    model = Filter(2, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=parser.Learning_rate)
    xy = list()
    z = list()
    result = list()
    
    # iterate filter
    for i in range(parser.filter_size):
        for j in range(parser.filter_size):
            xy.append([i,j])
            z.append(input[i,j])
            
    #training data 一起訓練的原因 : 讓部分突出的回歸線被濾掉 (弱化學習)
    for epoch in range(parser.epoch):
        feature = torch.Tensor(xy).to(device)
        target = torch.Tensor(z)[:,None].to(device)
        
        # forward pass and loss
        y_predicted = model(feature)
        loss = criterion(y_predicted, target)
        
        # backward pass
        loss.backward()

        # update
        optimizer.step()
        
        
    #create param to numpy
    for param in model.parameters(): # a b bias
        l = param.detach().view(1,-1).tolist()[0]
        for i in l:
            result.append(i)
    
    arr[tar_x,tar_y,:] = np.array(result)

# demo 3D regression輸出的結果
def draw(t,file_name,RGBIMAGE,parser):
    t = cv2.resize(t,(512,256)).astype('float32')
    # RGB = cv2.cvtColor(RGBIMAGE, cv2.COLOR_RGB2GRAY)
    # RGB = np.stack((RGB,)*3, axis=-1)
    RGBIMAGE = cv2.cvtColor(RGBIMAGE, cv2.COLOR_RGB2BGR)
    tonemapDurand = cv2.createTonemap(gamma = 1.5)
    # RGB = tonemapDurand.process(RGB)
    # cv2.imshow('light',RGBIMAGE)
    # cv2.imshow('light2',t)
    # t[:,:,0] = t[:,:,0] + RGB
    # t[:,:,1] = t[:,:,1] + RGB
    # t[:,:,2] = t[:,:,2] + RGB 
    dst = cv2.addWeighted(t,1.0,RGBIMAGE,0,0)
    # 直接轉換成jpg 然後clip 的filter會比tonemapping 的好
    t = np.clip(tonemapDurand.process(t) * 255, 0, 255).astype('uint8')  
    # cv2.imwrite(f'{parser.tend_output}/{file_name}.jpg', t)
    cv2.imwrite(f'{parser.tend_output}/{file_name}.jpg', t) 
    # cv2.imshow('light3',t)   
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    
def start(parser):
    dir = list()
    for ext in parser.extension:
        dir.extend(glob.glob(parser.source + f'/*{ext}'))
        print(parser.source + f'/*{ext}')
    print(f"Find source file {dir} len {len(dir)}")
    assert len(dir)!=0,'Direction is empty or parser of File type is incorrect'
    print(f"running on {parser.device}")
    random.shuffle(dir)
    start_time = time.time()
    for i in tqdm(dir):
        finish_dir = glob.glob(parser.tend_output+'/*.jpg')
        names = [os.path.basename(x).split('.')[0].split('_')[0] for x in finish_dir]
        if os.path.basename(i).split('.')[0] in names:
            print(f"\n{i} exists...skip")
            continue
        #read image
        color = cv2.imread(i,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        RGBimage = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) #BGR to RGB
        RGBimage = cv2.resize(RGBimage,(512, 256), interpolation=cv2.INTER_AREA)
        if parser.source_type == 'LDR':
            RGBimage = RGBimage/255
        shape = RGBimage.shape
        print(f'\n{i} shape : {shape}')
        
        #create final matrix
        Strength_image = np.zeros((shape[0],shape[1]))
        Strength_image = RGBimage[:,:,0]*0.299 +  RGBimage[:,:,1]*0.587+ RGBimage[:,:,2]*0.114
        
        # assertion
        if shape[0]%parser.filter_size!=0 or shape[1]%parser.filter_size!=0:
            print(f"stride problem happen at {i}")
            continue
        
        tend = np.zeros((int(shape[0]/parser.filter_size),int(shape[1]/parser.filter_size),3)) # a , b , bias => y = ax+by+bias
        
        #setting
        global device
        global criterion
        device = torch.device(parser.device)
        criterion = nn.MSELoss()
        #torch.manual_seed(0) # 指定網路initial parameters
        
        #iterate image
        for j in tqdm(range(0,shape[0],parser.filter_size)):
            for k in range(0,shape[1],parser.filter_size):
                train(tend,int(j/parser.filter_size),int(k/parser.filter_size),parser,Strength_image[j:j+parser.filter_size,k:k+parser.filter_size])
        file_name = Path(i).stem
        draw(tend,file_name,RGBimage,parser) #draw for demo parameter
    
    elapsed_time = time.time() - start_time
    print(f"Time cost {elapsed_time}")

    
if __name__ == '__main__':
    parser = create_parser()
    start(parser)