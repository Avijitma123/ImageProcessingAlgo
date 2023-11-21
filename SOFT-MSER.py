from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

import cv2
from PIL import Image
from numpy import asarray

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
import cv2
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import tensorflow.keras
from PIL import Image, ImageOps

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#===========================================================================================================
img1 = cv2.imread("Test6.jpeg")
#Convert the input imge into grayscale image 
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
d = img.shape
#Component list
inv_component=[]
norm_component=[]
print(d)
      
def component_find(k,I):
    l=[]
    for i in range(int(d[0])):
        for j in range(int(d[1])):
            if(I[i][j]==k):
                l.append((i,j))
    return l 
def findElement(key,mat):
    l=[]
    # base case
    #if not mat or not len(mat):
         #return
 
    # `M × N` matrix
    (M, N) = (len(mat), len(mat[0]))
    
    # start from `(0, N-1)`, i.e., top-rightmost cell of the matrix
    i = 0
    j = N - 1
 
    # run till matrix boundary is reached
    while i <= M - 1 and j >= 0:
 
        # if the current element is less than the key, increment row index
        if mat[i][j] < key:
            i = i + 1
 
        # if the current element is more than the key, decrement col index
        elif mat[i][j] > key:
            j = j - 1
 
        # if the current element is equal to the key
        elif mat[i][j]==key:
            l.append((i,j))
            i = i + 1
            j = j - 1  
            
    return l 
def Find_area(key,j):
        ret, thresh = cv2.threshold(img, j, 255, cv2.THRESH_BINARY_INV)
        
        connectivity = 8
        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        area = stats[key, cv2.CC_STAT_AREA]
        return area
        
    
                 
def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count 


centa={}
#Compute Difference between two soft set.    
def difference(Soft1,Soft2,t1,t2):
    p={}
    x=[v for k,v in Soft1.items()]
    y=[v for k,v in Soft2.items()]
    key1=[]
    key2=[]
    for k,v in Soft1.items():
            
            sp1=k.split("=")[1]
            tp1=sp1.split(",")
            Cx=tp1[0]
            Cy=tp1[1]
            
            key1.append((Cx,Cy))
    
    for k,v in Soft2.items():
            
            sp2=k.split("=")[1]
            tp2=sp2.split(",")
            Cx=tp2[0]
            Cy=tp2[1]
            
            
            key2.append((Cx,Cy))        
    if(len(x)>len(y)):
          for dp in key1:
              if dp not in key2:
                    key1.remove(dp)
    elif(len(x)<len(y)): 
          for dp in key2:
              if dp not in key1:
                    key2.remove(dp)
    #print("Key1--->",len(key1))
    #print("Key2--->",len(key2))
    x_e=[]
    y_e=[]
    for k,v in Soft1.items():
            
            sp1=k.split("=")[1]
            tp1=sp1.split(",")
            Cx=tp1[0]
            Cy=tp1[1]
            if (Cx,Cy) in key1:
                x_e.append(((Cx,Cy),int(v)))
    for k,v in Soft2.items():
            
            sp1=k.split("=")[1]
            tp1=sp1.split(",")
            Cx=tp1[0]
            Cy=tp1[1]
            if (Cx,Cy) in key2:
                y_e.append(((Cx,Cy),int(v)))            
    #print("X_e-->",len(x_e))
    #print("Y_e-->",len(y_e))
    #print(str(t1)+"-"+str(t2))
    Y_e={}
    for k in range(len(y_e)):
        cent=y_e[k][0]
        val=y_e[k][1]
        Y_e[cent]=val
    prev=set(x_e)
    curr=set(y_e)
    #print("check---->",prev.difference(curr))
    
            
    s=str(t1)+"-"+str(t2) 
    s_c=[]                                     
    for i in range(len(x_e)):
         q_x=x_e[i][0]
         
         size_q_x=x_e[i][1]
         if q_x in Y_e.keys():
              d=size_q_x - Y_e[q_x]
              print(s,d)
              s_c.append((q_x,d))
             
         
                   
    centa[s]=s_c             
                    
                      
print(centa)               
                    
def Info_mat(img):
    data={} 
        
    for j in range(120,200):
        label_set={} 
        ret, thresh = cv2.threshold(img, j, 255, cv2.THRESH_BINARY_INV)
        print("For t=", j)
    

        connectivity = 8
        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        
        for i in range(0,numLabels):
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            
            
            #print("size--->",out)
            st=str(i)+"="+str(int(cX))+","+str(int(cX))
            label_set[st]=Find_area(i,j)
        data[j]=label_set
    return data
        
                                                
              
                      
data=Info_mat(img)
print(data)
keys_set=[k for k in data.keys()]
set1=[]
for i in range(len(keys_set)-1):
    t1=keys_set[i]
    t2=keys_set[i+1]
    difference(data[t1],data[t2],t1,t2)  


#df3=pd.DataFrame.from_dict(set1)
def Count_zero(l):
    count=0
    for i in range(len(l)):
        k=l[i]
        if(k[1]==0):
            count=count+1
    return count        
        



zero_count={}
len_count={}
for k,v in centa.items():
    c=Count_zero(v)
    sp=len(v)
    len_count[k]=sp
    zero_count[k]=c
print( "Zeros--->",zero_count) 
print("================================================================")
print(len_count)  
def Component_marking(th):
    ret,thresh=cv2.threshold(img, th, 255,cv2.THRESH_BINARY)
    connectivity=8
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    
    output = img1.copy()
    k=255
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        (cX, cY) = centroids[i]
        
        if(area > 50):
             (cX, cY) = centroids[i]
             norm_component.append(((cX,cY),area))
             cv2.rectangle(output, (x, y), (x + w, y + h), (k, 0, 0), 3)
             #cv2.circle(output, (int(cX), int(cY)), 4, (s, 0, k), -1)
             componentMask = (labels == i).astype("uint8") * 255
             # show our output image and connected component mask
             #cv2.imshow("Output", output)
             #print(componentMask)
             #cv2.imshow("Connected Component", componentMask)
             output=output.copy()
             k=k-50
             cv2.waitKey(0)
    return output  
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows() 
         
def Component_marking_inv(th):
    I=Component_marking(th)
    ret,thresh=cv2.threshold(img, th, 255,cv2.THRESH_BINARY_INV)
    connectivity=8
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    
    output = I.copy()
    k=255
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        if(area >50):
            
             inv_component.append(((cX,cY),area))
             cv2.rectangle(output, (x, y), (x + w, y + h), (k, 0, 0), 3)
             #cv2.circle(output, (int(cX), int(cY)), 4, (s, 0, k), -1)
             componentMask = (labels == i).astype("uint8") * 255
             # show our output image and connected component mask
             #cv2.imshow("Output", output)
             #print(componentMask)
             #cv2.imshow("Connected Component", componentMask)
             output=output.copy()
             k=k-50
             cv2.waitKey(0)
      
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()          
all_values = zero_count. values()
max_value = int(sum(all_values) / len(all_values))
component=len_count.values()
avg=int(sum(component)/len(component))

#sp=find_avg_loc(avg)
print(max_value) 
#p=sp.split("-") 
T=0
for k,v in zero_count.items():
      st=k.split("-")
      if(v<=max_value+4 and v>=max_value-4):
            
            print("T---->",st[0])
            T=int(st[0])
            Component_marking_inv(int(st[0]))  
            break
"""print("For normal case: ",len(norm_component))
print("For INV case: ",len(inv_component))"""
                 
#========================================Component collector part========================================
model = tensorflow.keras.models.load_model('model_version_2.h5')
ks=int(d[0])*int(d[1])
i_component=[]
nor_component=[]
def Component_marking(th):
    ret,thresh=cv2.threshold(img, th, 255,cv2.THRESH_BINARY)
    connectivity=8
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    
    output = img1.copy()
    k=255
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        (cX, cY) = centroids[i]
        im=img1[y:y+h+4,x:x+w+4]
        resized = cv2.resize(im, (224,224), interpolation = cv2.INTER_AREA)
        
        img_array = image.img_to_array(resized )
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        p=model.predict(img_preprocessed)
        if(area > 50 and w!=d[1] and h!=d[0] and p==0):
             (cX, cY) = centroids[i]
             norm_component.append([x,y,w,h,area])
             cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 3)
             #cv2.circle(output, (int(cX), int(cY)), 4, (255, 0, 0), -1)
             componentMask = (labels == i).astype("uint8") * 255
             # show our output image and connected component mask
             cv2.imshow("Output", output)
             #print(componentMask)
             #cv2.imshow("Connected Component", componentMask)
             output=output.copy()
             #k=k-50
             cv2.waitKey(0)
    return output  
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows() 
         
def Component_marking_inv(th):
    I=Component_marking(th)
    ret,thresh=cv2.threshold(img, th, 255,cv2.THRESH_BINARY_INV)
    connectivity=8
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    
    output = I.copy()
    k=255
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        im=I[y:y+h+4,x:x+w+4]
        resized = cv2.resize(im, (224,224), interpolation = cv2.INTER_AREA)
        
        img_array = image.img_to_array(resized )
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        p=model.predict(img_preprocessed)
        
        
        (cX, cY) = centroids[i]
        if(area >50 and w!=d[1] and h!=d[0] and p==0):
             
             inv_component.append([x,y,w,h,area])
             cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 3)
             #cv2.circle(output, (int(cX), int(cY)), 4, (s, 0, k), -1)
             componentMask = (labels == i).astype("uint8") * 255
             # show our output image and connected component mask
             cv2.imshow("Output", output)
             #print(componentMask)
             #cv2.imshow("Connected Component", componentMask)
             output=output.copy()
             #k=k-50
             cv2.waitKey(0)
      
    if cv2.waitKey(0) & 0xff == 27:
        filename="datafinal2"+".jpg"
        cv2.imwrite(filename, output)
        cv2.destroyAllWindows() 
Component_marking_inv(T)
print(inv_component)
print(norm_component)   
          
     

      

     
         
