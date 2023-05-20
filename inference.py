import os
import Scripts.LDR2HDR.LDR2HDRinference as L2H
import Scripts.light_detector.Median_cut as MC
import Scripts.light_detector.Filter_pytorch as FP
import Scripts.light_detector.Yolo as Yolo
import Scripts.Depth.Trans_depth as TD
import glob
import time
import shutil

def clear():
    delete_file = glob.glob(f'./*/*/*/*.jpg')
    delete_file.extend(glob.glob(f'./*/*/*/*.exr'))
    delete_file.extend(glob.glob(f'./*/*/*/*.npy'))
    delete_file.extend(glob.glob(f'./*/*/*/*.json'))
    delete_file.extend(glob.glob(f'./*/*/*/*.hdr'))
    for i in delete_file:
        os.remove(i) 
    print(f'finish clearing file:\n {delete_file}\n')

# Start full program
def start():
    base_time = time.time() # count for total time
    args_FP,remain= FP.set_parser().parse_known_args()
    args_L2H,_ = L2H.parse_arguments(remain)
    #Setting file type 
    if(args_FP.source_type == 'HDR'):
        args_FP.extension = ['.exr','.hdr']
    elif(args_FP.source_type == 'LDR'):
        args_FP.extension = ['.jpg','png']
    print(args_FP)
    print(args_L2H)
    L2H.main(args_L2H)
    TD.start()
    FP.start(args_FP)
    Yolo.start()
    print(f"Total time cost{time.time()-base_time}")
    shutil.copy2('./Scripts/LDR2HDR/HDR_nolight/upload_nolight.hdr', f"./output/output.hdr")
    shutil.copy2('./Scripts/LDR2HDR/images_LDR/upload.jpg', f"./output/origin.jpg")
    shutil.copy2('./Scripts/LDR2HDR/images_HDR/upload.exr', f"./output/Ldr2hdr.exr")
    shutil.copy2('./Scripts/Depth/image_depth/upload.npy', f"./output/output.npy")
    shutil.copy2('./Scripts/Depth/image_depth/upload_depth.jpg', f"./output/depth.jpg")
    shutil.copy2('./Scripts/light_detector/to_predict/upload.jpg', f"./output/gradient.jpg")
    shutil.copy2('./Scripts/light_detector/data/upload.json', f"./output/output.json")
    shutil.copy2('./Scripts/light_detector/prediction_view/upload.jpg', f"./output/detect.jpg")
    pass

if __name__ == '__main__':
    input= input("Clean old: ")
    if input!='N' and input!='No':
        clear()
    finish_dir = glob.glob('input/*.jpg')
    for i in finish_dir:
        shutil.copy2(i, f"./Scripts/LDR2HDR/images_LDR/{os.path.basename(i)}")
    start()