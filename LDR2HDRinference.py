import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import math
import glob
import sys
from pathlib import Path
import numpy as np
import torch
#import torchvision
import time
from tqdm import tqdm
from LDR2HDR_loaders.Illum_loader import IlluminationModule, Inference_Data
from LDR2HDR_loaders.autoenc_ldr2hdr import LDR2HDR    
from torch.utils.data import DataLoader

def get_alpha(a,epi):
    b = np.copy(a)
    for i in range(0,3):
        for v in range(0,256):
            for z in range(0,512):
                b[i][v][z] = max(0,max(a[2][v][z],max(a[0][v][z],a[1][v][z]))-epi)/(1-epi)    
    return b

def get_H_nolight(H,alpha,input_ldr,tar,gamma):
    exp = np.copy(H)
    g = np.copy(H)
    for i in range(0,3):
        for v in range(0,256):
            for z in range(0,512):
                H[i][v][z] = (1-alpha[i][v][z])* math.pow(input_ldr[i][v][z],gamma)+  alpha[i][v][z]* (1/math.exp(tar[i][v][z]))
                exp[i][v][z] = math.exp(tar[i][v][z])
                g[i][v][z] = math.pow(input_ldr[i][v][z],gamma)
    return H,exp,g
def get_H_origin(H,alpha,input_ldr,tar,gamma):
    exp = np.copy(H)
    g = np.copy(H)
    for i in range(0,3):
        for v in range(0,256):
            for z in range(0,512):
                H[i][v][z] = (1-alpha[i][v][z])* math.pow(input_ldr[i][v][z],gamma)+ alpha[i][v][z]* math.exp(tar[i][v][z])
                exp[i][v][z] = math.exp(tar[i][v][z])
                g[i][v][z] = math.pow(input_ldr[i][v][z],gamma)
    return H,exp,g

def parse_arguments(args):
    usage_text = (
        "Inference script for Deep Lighting Environment Map Estimation from Spherical Panoramas"
        "Usage:  python3 inference.py --input_path "
    )
    parser = argparse.ArgumentParser(description=usage_text)    
    parser.add_argument('--input_path', type=str, default='./Loader/images_LDR', help="Input panorama color image file")
    parser.add_argument('--out_path', type=str, default='./Loader/images_HDR', help='Output folder for the predicted environment map panorama')
    parser.add_argument('-g','--gpu', type=str, default='0', help='GPU id of the device to use. Use -1 for CPU.')    
    parser.add_argument('--ldr2hdr_model', type=str, default='./models/ldr2hdr.pth', help='Pre-trained checkpoint file for ldr2hdr image translation module')
    parser.add_argument("--width", type=float, default=512, help = "Spherical panorama image width.")
    parser.add_argument('--deringing', type=int, default=0, help='Enable low pass deringing filter for the predicted SH coefficients')    
    parser.add_argument('--dr_window', type=float, default='6.0')
    return parser.parse_known_args(args)

def evaluate(
    ldr2hdr_module: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device
):
    if (os.path.isdir(args.out_path)!=True):
        os.mkdir(args.out_path)
    dr = list()
    dr.extend(glob.glob(args.input_path + f'/*jpg'))
    print(dr)
    for i in tqdm(dr):
        finish_dir = glob.glob(f'{args.out_path}/*.exr')
        names = [os.path.basename(x).split('.')[0] for x in finish_dir]
        if os.path.basename(i).split('.')[0] in names:
            print(f"\n{i} exists...skip")
            continue
        p = Path(i)
        in_filename, in_file_extention = p.stem,p.suffix
        print(in_filename)
        assert in_file_extention in ['.png','.jpg']
        inference_data = Inference_Data(i)
        out_path = args.out_path + os.path.basename(i)
        out_filename, out_file_extension = os.path.splitext(out_path)
        out_file_extension = '.exr'
        out_path = out_filename + out_file_extension
        dataloader = DataLoader(inference_data, batch_size=1, shuffle = False, num_workers = 1)
        
        for i, data in enumerate(dataloader): # enumerate 數 list 裡面第幾個元素 
            input_img = data.to(device).float() #讀入img到cpu or gpu
            input_ldr = input_img.cpu().detach().numpy()[0] #因為numpy不能在gpu理運作所以要將ldr拉到cpu
            #print(input_ldr.dtype)
            with torch.no_grad(): 
                start_time = time.time()
                right_rgb = ldr2hdr_module(input_img)
                epi = 0.9#改這個是改變 gamma correction 和 HDR output domain 的 blending 比值
                gamma = 2 #gamma correction (要先將範圍normalize 到 0,1)
                tmp = right_rgb[0].cpu().detach().numpy()
                alpha = get_alpha(np.copy(input_ldr),epi)
                # cv2.imwrite(f'Alpha.exr',np.transpose(alpha,(1,2,0)).astype(np.float32))
                H = np.empty([3,256,512])
                H_origin = np.empty([3,256,512])
                # dark = np.transpose(tmp,(1,2,0)).astype(np.float32)
                # cv2.imwrite('dark.exr', dark)
                H_origin,exp,g = get_H_origin(H_origin,alpha,input_ldr,tmp,gamma)
                H,exp,g = get_H_nolight(H,alpha,input_ldr,tmp,gamma)
                #cv2.imwrite('exp_y_hat.exr',np.transpose(exp,(1,2,0)).astype(np.float32))
                #cv2.imwrite('gamma.exr',np.transpose(g,(1,2,0)).astype(np.float32))
                H = np.transpose(H,(1,2,0)).astype(np.float32)
                H_origin = np.transpose(H_origin,(1,2,0)).astype(np.float32)
                cv2.imwrite(f'./HDR_nolight/{in_filename}_nolight.exr', H)
                cv2.imwrite(f'{args.out_path}/{in_filename}.exr', H_origin)
                H = np.transpose(H,(2,0,1)).astype(np.float32)
                # alpha_H = get_alpha(np.copy(H),epi)
                # cv2.imwrite('Alpha_no_light.exr',np.transpose(alpha_H,(1,2,0)).astype(np.float32))
            
def main(args):
    device = torch.device("cuda:" + str(args.gpu) if (torch.cuda.is_available() and int(args.gpu) >= 0) else "cpu")    
    print(device)
    # load LDR2HDR module     
    ldr2hdr_module = LDR2HDR()
    ldr2hdr_module.load_state_dict(torch.load(args.ldr2hdr_model, map_location=torch.device('cpu'))['state_dict_G'])
    ldr2hdr_module = ldr2hdr_module.to(device)
    print("LDR2HDR moduled loaded")
    evaluate(ldr2hdr_module, args, device)
    
if __name__ == '__main__':
    args, unknown = parse_arguments(sys.argv)
    main(args)