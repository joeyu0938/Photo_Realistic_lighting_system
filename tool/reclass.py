import glob
import os
from tqdm import tqdm
path = 'C:/Users/User/Desktop/yolov7-main_test/data/lights/labels/'
dir = []
dir.extend(glob.glob(f'{path}*/*.txt'))
print(dir)
for i in tqdm(dir):
    f = open(i,'r+')
    word = []
    cnt = 0
    for line in f.readlines():
        if(line[0]=='2'):
            tmp ='1'+line[1:]
            word.append(tmp)
            cnt+=1
        elif(line[0]=='3'):
            tmp ='0'+line[1:]
            word.append(tmp)
            cnt+=1
        else:
            word.append(line)
        pass
    f.seek(0)
    f.truncate(0)
    f.writelines(word)
    f.close
    if cnt>0:
        print(f"modified {i}\n")