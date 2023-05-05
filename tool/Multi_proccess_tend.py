# 多個process 進行
import os 
import sys
import tool.Filter_pytorch as fp
import multiprocessing as mp
def start(q):
    p_list = []
    
    print(os.cpu_count())
    for i in range(0,8): # os.cpu_count-1 
        p_list.append(mp.Process(target=fp.start,args=(q,)))
        
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()
    print("Done")
    
if __name__ == '__main__':
    args = fp.create_parser()
    start(args)