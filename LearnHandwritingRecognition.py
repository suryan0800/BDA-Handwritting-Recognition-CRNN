# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:12:09 2020

@author: surya
"""
#Testing CV2

import cv2 
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
import numpy as np
import os

os.chdir('C:\\Users\\surya\\Documents\\Lab\\Python\\BigDataProject')
fle = "HandwrittenDataset/test_v2/test/TEST_0006.jpg"
#fle = "HandwrittenDataset/Hello.jpg"
#fle = "HandwrittenDataset/Chandra1.jpg"
img = cv2.imread(fle, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(256,64))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
display(Image.fromarray(img))
print(img)

#clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#cl = clahe.apply(img)
#display(Image.fromarray(cl))

img2 = img.flatten()


plt.hist(img2)
plt.title("Histogram")
plt.xlabel("GrayScale Value")
plt.ylabel("Frequency")
plt.show()


#plt.hist(th3.flatten())
for i in range(len(img2)):
    if img2[i] <= 120:
        img2[i] = 0  
    #elif img2[i] > 130:
    #    img2[i] = 254
    else:
        img2[i] = 254
        
img3 = np.reshape(img2,img.shape)
display(Image.fromarray(img3))

cv2.imshow("image",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
display(Image.fromarray(thresh1))

contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 



for cnt in contours: 
    img4 = thresh1.copy()
    x, y, w, h = cv2.boundingRect(cnt) 
    start_point = (x,y) 
    end_point = (x+w,y+h) 
    color = (255, 0, 0) 
    thickness = 2
    img4 = cv2.rectangle(img4, start_point, end_point, color, thickness) 
  
    display(Image.fromarray(img4))

'''
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

img = cv2.imread("Ad.jpg")
text = (pytesseract.image_to_string(img))
'''