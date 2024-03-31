#Importing required Libraries
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
#import matplotlib.pyplot as plt

#Mapping(can be directly imported from model.ipynb)
label_map = pd.read_csv("./data/emnist-balanced-mapping.txt",
                        delimiter = ' ',
                        index_col=0,
                        header=None).squeeze()
label_dictionary = {}
for index, label in enumerate(label_map):
    label_dictionary[index] = chr(label)

def preprocessing(input_image):
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    ret, im_th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_th = cv2.adaptiveThreshold(im_th, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
    edge_image = cv2.Canny(im_th, 0, 255)
    return edge_image


img = cv2.imread('./test_img2.jpg')

#Contour and canny edge detection
edged = preprocessing(img)
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in contours]

#Group bounding rectangles of letters in a line
dict_ordered_by_y = {}
def insert_letter(rect):
    for key in dict_ordered_by_y.keys():
        if rect[1]>=dict_ordered_by_y[key][0][0] and rect[1]+rect[3]<=dict_ordered_by_y[key][0][1]:
            dict_ordered_by_y[key].append((rect[0],rect[1],rect[2],rect[3]))
            dict_ordered_by_y[key][0][0]=min(dict_ordered_by_y[key][0][0],max(0,rect[1]-(rect[3])/3))
            dict_ordered_by_y[key][0][1]=min(dict_ordered_by_y[key][0][1],rect[1]+rect[3]+(rect[3])/3)
            return True
    dict_ordered_by_y[(rect[0],rect[1],rect[2],rect[3])]=[[max(0,rect[1]-(rect[3])/3),rect[1]+rect[3]+(rect[3])/3]]
    return False

for r in rects:
  insert_letter(r)

#Sort Letters by x-coordinate
letters=[]
for key,val in dict_ordered_by_y.items():
  l=[]
  l.append(key)
  for i in range(1,len(val)):
    l.append(val[i])
  l.sort()
  letters.append(l)
letters.sort()

#Load model
class ICR(nn.Module):
    def __init__(self):
        super(ICR,self).__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*6*6,512),
            nn.Dropout(0.5),
            nn.Linear(512,47)
        )
    def forward(self,X):
        out=self.layers(X)
        return out

model = torch.load('./charClassifier.pth')
model.eval()

#Predict results
result=""
for it in letters:
    prev=-1
    for index,rect in enumerate(it):
      roi = img[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
      roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
      roi=np.pad(roi,pad_width=3,constant_values=255)
      roi = cv2.resize(255-roi, (28, 28))
      roi=cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
      roi=roi.astype('float32') / 255

      image = roi.astype(np.float32)
      image=image.reshape(-1,1,28, 28)
      image=torch.from_numpy(image)
      prediction =model(image)
      prediction=np.argmax(prediction.detach().numpy(),axis=-1)
      if prev!=-1:
        if prev[0]+1.5*prev[2]<rect[0]:
          result+=" "
      result+=label_dictionary[prediction[0]]
      prev=rect
    result+='\n'

#Trick to remove zeros from words
resultlist=list(result)
for i in range(0,len(resultlist)):
   if resultlist[i]=='0' and (i==0 or resultlist[i-1].isalpha() or resultlist[i-1]==" ") and (i==len(resultlist)-1 or resultlist[i+1].isalpha() or resultlist[i+1]==" "):
      resultlist[i]="O"
result=''.join(resultlist)

print(result)