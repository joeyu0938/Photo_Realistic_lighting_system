#用來檢查 label txt 和 image hdr 的數量有沒有一致
import glob
import os
dir = []
x = []
x.extend(glob.glob('C:/Users/User/Desktop/Trainig Dataset/Label/*txt'))
dir.extend(glob.glob('C:/Users/User/Desktop/yolov7-main_test/data/lights/tends/train/*npy'))
names = [os.path.basename(y).split('.')[0] for y in x]
cnt = 0
for i in dir:
    if os.path.basename(i).split('.')[0] not in names:
        #os.remove(i) 
        print(f"\n{i} is superfluous...skip")
        cnt+=1
        continue
print(f"Missing {cnt} files")