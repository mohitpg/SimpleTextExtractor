import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

label_map = pd.read_csv("./data/emnist-balanced-mapping.txt",
                        delimiter = ' ',
                        index_col=0,
                        header=None).squeeze()
label_dictionary = {}
for index, label in enumerate(label_map):
    label_dictionary[index] = chr(label)

def preprocessing(input_image, edge=False, inv_thresh=False):
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    if inv_thresh:
        ret, im_th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        im_th = cv2.adaptiveThreshold(im_th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 10)
        im_th = cv2.bitwise_not(im_th)
    else:
        ret, im_th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im_th = cv2.adaptiveThreshold(im_th, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
    if edge:
        edge_image = cv2.Canny(im_th, 0, 255)
        return edge_image
    return im_th


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

img = cv2.imread('./test_img1.jpg')
edged = preprocessing(img, edge=True,inv_thresh=False)

contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in contours]
d={}
def inserter(rect):
  for key in d.keys():
    if rect[1]>=d[key][0][0] and rect[1]+rect[3]<=d[key][0][1]:
      d[key].append((rect[0],rect[1],rect[2],rect[3]))
      d[key][0][0]=min(d[key][0][0],max(0,rect[1]-(rect[3])/3))
      d[key][0][1]=min(d[key][0][1],rect[1]+rect[3]+(rect[3])/3)
      return True
  d[(rect[0],rect[1],rect[2],rect[3])]=[[max(0,rect[1]-(rect[3])/3),rect[1]+rect[3]+(rect[3])/3]]
  return False

for r in rects:
  inserter(r)
print(d)
l=[]
for key,val in d.items():
  l1=[]
  l1.append(key)
  for i in range(1,len(val)):
    l1.append(val[i])
  l1.sort()
  l.append(l1)
l.sort()


result = ''
print(result)
model = torch.load('./charClassifier.pth')
model.eval()
#img = cv2.imread('/content/test2.jpg')
result=""
for it in l:
    prev=-1
    for index,rect in enumerate(it):
      roi = img[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
      #print(rect)
      gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
      gray=np.pad(gray,pad_width=3,constant_values=255)
      gray = cv2.resize(255-gray, (28, 28))
      gray=cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
      gray=gray.astype('float32') / 255
      roi=gray
      #roi = preprocessing(roi)

      #image = cv2.resize(roi, (28, 28))
      image = roi.astype(np.float32)
      image=image.reshape(-1,1,28, 28)
      image=torch.from_numpy(image)
      prediction =model(image)
      #print(prediction)
      prediction=np.argmax(prediction.detach().numpy(),axis=-1)
      # plt.imshow(gray,cmap=plt.cm.gray)
      # plt.title(label_dictionary[prediction[0]])
      # plt.show()
      if prev!=-1:
        if prev[0]+1.5*prev[2]<rect[0]:
          result+=" "
      result+=label_dictionary[prediction[0]]
      prev=rect
    result+='\n'

print(result)