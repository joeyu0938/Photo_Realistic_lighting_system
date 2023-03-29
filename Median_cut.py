import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import json
import cv2
final_place = list()
final_area = list()
final_point = list()
# 最後的輸出 光源的位置 一個框框配一個 box_type class
count_box = list()
LOW_LIGHT = 0.1
#input a,b,c :  RGB float
def lum(a,b,c):
    #return max(0,max(a,max(b,c)-0.95)/(1-0.95))# 用inference 跑出alpha 的公式 除法和減法似乎沒差
    return a*0.114 + b*0.587 + c*0.299 #可以改變 

# Summed table 
class sum_t:
    def __init__(self,start,end,height,width,arr):
        self.I = np.ones((height,width)) # I 是這個table class 的變數
        for i in range(start,start+height):
            for j in range(end,end+width):
                value = lum(arr[0][i][j],arr[1][i][j],arr[2][i][j])
                if(i>start):
                    value += self.I[i-start-1][j-end]
                if(j>end):
                    value += self.I[i-start][j-end-1]
                if(i>start and j>end):
                    value -= self.I[i-start-1][j-end-1]
                self.I[i-start][j-end] = value

#  要切割的區域 
class sat_r:
    # sum 是summed table
    def __init__(self,x,y,height,width,sum):#已經減一的height
        self.area = sum[x+height][y+width] -sum[x+height][y]-sum[x][y+width] + sum[x][y]
        self.height = height
        self.width = width
        self.x = x
        self.y = y
    def split_h(self,sum):
        for i in range(0,self.height+1):
            y = sat_r(self.x,self.y,i,self.width,sum)
            if(y.area*2 >= self.area):
                b = sat_r(self.x+i,self.y,self.height-i,self.width,sum)
                self.split_point_h = self.x+i
                break
        return y,b
    def split_w(self,sum):
        for i in range(0,self.width+1):
            y = sat_r(self.x,self.y,self.height,i,sum)
            if(y.area*2 >= self.area):#有無法成立的情況
                b = sat_r(self.x,self.y+i,self.height,self.width-i,sum)
                self.split_point_w = self.y+i
                break
        return y,b
    def centroid(self,sum):
        self.split_h(sum)
        self.split_w(sum)
        return self.split_point_h,self.split_point_w
    
# 框框的class 
class box_type:
    def __init__(self,type):
        # accum 所有加入過的光 按照遞回次數排序
        self.accum = list()
        # bright 代表最亮的點 如果是point light 就可以直接存取
        self.bright = [0.0,0.0,0.0] #最強的 x y 光強度
        # x_y 所有加入過的光 不按照遞回次數排序
        self.x_y = list()
        self.type = type
        pass
    # level 是框框遞回的深度
    def add(self,arr,level):
        if(level>=len(self.accum)):
            new = list()
            self.accum.append(new)
        numbers = arr.copy()
        #判斷光源是否太近 太近就不理會
        if(numbers not in self.x_y and [numbers[0]-1,numbers[1]-1]  not in self.x_y and [numbers[0],numbers[1]-1]  not in self.x_y and [numbers[0]-1,numbers[1]]  not in self.x_y):
            self.x_y.append(numbers)
            l = lum(color[0][arr[0]][arr[1]],color[1][arr[0]][arr[1]],color[2][arr[0]][arr[1]])
            arr.append(l)
            if(l>self.bright[2]):
                self.bright = arr
            self.accum[level].append(arr)
            
# times 紀錄遞回次數 , start: 原圖框框最左上的x  , end:原圖框框最左上的y , sat : 切割出的區域 , sum: summed table, limit: 深度限制 , box_id   
def recursive(times,start,end,sat:sat_r,sum,limit,box_id):
    if(times>limit):
        return
    x,y = sat.centroid(sum)
    x+=start
    y+=end
    count_box[box_id].add([x,y],times) 
    #cv2.circle(origin2, center=(y, x), radius=0, color=(0, 255, 0))
    final_place.append([x,y])
    if(sat.height>sat.width):
        A_,B_ = sat.split_h(sum)
    else:
        A_,B_ = sat.split_w(sum)
    recursive(times+1,start,end,A_,sum,limit,box_id)
    recursive(times+1,start,end,B_,sum,limit,box_id)

#return light :json
# {
#     "type":
#     "position":
#     "color":
#     "intensity":
# }

# label_arr: label  0:y  1:x  2:width  3:height
def medium_cut(image_path,label_arr,debug_LDR_path=""):
    count_box.clear()
    final_point.clear()
    final_area.clear()
    final_place.clear()
    global color
    color = cv2.imread(image_path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    color = np.transpose(color,(2,0,1))
    #只讓光源區域進行median_cut cnt 是紀錄第幾個框框 
    cnt=0
    if(len(debug_LDR_path)!=0):
        origin = cv2.imread(debug_LDR_path, cv2.IMREAD_UNCHANGED)
        origin2 = cv2.resize(origin, (512,256), interpolation=cv2.INTER_CUBIC)
        origin2 = np.transpose(origin2,(2,0,1))
    else:
        origin2 = color
    
    origin_image = np.transpose(origin2,(1,2,0))
    #cv2.imshow('origin', origin_image)
    
    label_nolabel = list()
    #光源的位置 一個框框配一個 box_type class
    for i in range(len(label_arr)):
        if(label_arr[i][0] == 0):
            count_box.append(box_type('Point'))
        else:
            count_box.append(box_type('Area'))
        label_nolabel.append(label_arr[i][1:])
    print(label_nolabel)
    light = list()
    final = dict()
    for i in label_nolabel:
         # 0:y 1:x 2:width 3:height
        summed_table = sum_t(i[1],i[0],i[3]+1,i[2]+1,color) # 將框框獨立出來做出summed table
        sat_origin = sat_r(0,0,i[3],i[2],summed_table.I) #  初始化切割區域
        recursive(0,i[1],i[0],sat_origin,summed_table.I,4,cnt)
        sat_image = origin_image[i[1]:i[1]+i[3]+1,i[0]:i[0]+i[2]+1]
        output = cv2.resize(sat_image, (1024,512))
        #cv2.imshow(f"{count_box[cnt].type}{cnt}",output)
        print(f"result{cnt} place from x: {i[1]} to {i[1]+i[3]+1} y:{i[0]} to {i[0]+i[2]+1} , same strength {len(final_place)}")
        #cv2.rectangle(origin_image,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,0,255),1)
        #cv2.putText(origin_image, count_box[cnt].type, (i[0],i[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (225,0,0), 0)
        if count_box[cnt].type == "Area":
            final_area.append(count_box[cnt].accum[3]) #取出遞回次數 3 的所有光源
            for v in count_box[cnt].accum[3]:
                #cv2.circle(origin_image, center=(v[1], v[0]), radius=0, color=(0, 0, 255))
                if(v[2]>LOW_LIGHT):
                    tar = dict()
                    tar["type"] = "Area"
                    tar["position"] = [v[0],v[1]] # x y
                    tar["color"] = list(reversed(origin2[:,v[0],v[1]].tolist())) # rgb 現在這裡是存 hdr 的rgb 0~1 如果想要用 LDR 就改成 origin_image 0~255
                    tar["intensity"] = v[2]
                    light.append(tar)
                    #cv2.circle(origin_image, center=(v[1], v[0]), radius=1, color=(0, 0, 255))
                else:
                    print("Delete dark light")
                pass
        else:
            if(count_box[cnt].bright[2]>LOW_LIGHT):
                final_point.append(count_box[cnt].bright) # 取出最亮的點
                tar = dict()
                tar["type"] = "point"
                tar["position"] = [count_box[cnt].bright[0],count_box[cnt].bright[1]] # x y
                tar["color"] = list(reversed(origin2[:,count_box[cnt].bright[0],count_box[cnt].bright[1]].tolist()))# rgb
                tar["intensity"] = count_box[cnt].bright[2]
                light.append(tar)
            else:
                print("Delete dark light")
            #cv2.circle(origin_image, center=(count_box[cnt].bright[1], count_box[cnt].bright[0]), radius=1, color=(0, 255, 0))
        output_full = cv2.resize(origin_image, (1024,512))
        # cv2.imshow('Result',output_full)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cnt+=1
    final["lightDataList"] = light
    # json dumps 把 list 的dict 全部轉換成 json 的 list 格式
    #print(json.dumps(light, indent=4)) # indent 可以拿掉 不然寫檔案速度會掉很慢
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent = 4)
    pass

#Read txt and ouput labeled array
def make_arr(path):
    arr = list()
    with open(path, 'r') as f:
        for line in f:
            tmp = line.split()
            list_of_ints = [float(x) for x in tmp]
            list_of_ints[3]*=512
            list_of_ints[4]*=256
            list_of_ints[1] = list_of_ints[1]*512-list_of_ints[3]/2
            list_of_ints[2] = list_of_ints[2]*256-list_of_ints[4]/2
            list_of_ints = [int(x) for x in list_of_ints]
            print(list_of_ints)
            arr.append(list_of_ints)

    return arr
    pass

if __name__ == '__main__':
    arr = make_arr("134.txt")
    medium_cut("134.exr",arr,"134.jpg")