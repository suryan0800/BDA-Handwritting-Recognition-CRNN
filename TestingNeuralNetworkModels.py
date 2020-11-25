# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:30:45 2020

@author: surya
"""



import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, log_loss
from IPython.display import display
from PIL import Image
import pylab as pl

os.chdir('C:\\Users\\surya\\Documents\\Lab\\Python\\BigDataProject')

valid = pd.read_csv("HandwrittenDataset/written_name_train_v2.csv")
#valid = pd.read_csv("HandwrittenDataset/written_name_validation_v2.csv")

valid.dropna(axis=0, inplace=True)
valid = valid[valid['IDENTITY'] != 'UNREADABLE']
valid['IDENTITY'] = valid['IDENTITY'].str.upper()
valid.reset_index(inplace = True, drop=True)



def preprocess(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
    #rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    #dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
    #display(Image.fromarray(thresh1))
    (h, w) = img.shape
    final_img = np.ones([64, 256])*0 # blank white image
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    final_img = final_img.astype("uint8")
    #display(Image.fromarray(final_img))
    final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    #display(Image.fromarray(final_img))
    return final_img


valid_size= 30000
start_ind = 60000
valid_x = []

for i in range(start_ind,start_ind+valid_size):
    #img_dir = 'HandwrittenDataset/validation_v2/validation/'+valid.loc[i, 'FILENAME']
    img_dir = 'HandwrittenDataset/train_v2/train/'+valid.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    image = preprocess(image)
    image = image/255.
    valid_x.append(image)

valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)




alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 34 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 64 # max length of predicted labels


def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret




valid_y = np.ones([valid_size, max_str_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])

for i in range(start_ind,start_ind + valid_size):
    valid_label_len[i-start_ind] = len(valid.loc[i, 'IDENTITY'])
    valid_y[i-start_ind, 0:len(valid.loc[i, 'IDENTITY'])]= label_to_num(valid.loc[i, 'IDENTITY'])    


#model = tf.keras.models.load_model("CNNModel")


preds = model.predict(valid_x)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)[0][0])

prediction = []
for i in range(valid_size):
    prediction.append(num_to_label(decoded[i]))



#model.save("CNNModel")





y_true = valid.loc[start_ind:start_ind+valid_size, 'IDENTITY']
y_true.reset_index(inplace = True, drop=True)
correct_char = 0
total_char = 0
correct = 0
y_true_char = []
y_pred_char = []
for i in range(valid_size):
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)
    
    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:
            correct_char += 1
        y_true_char.append(tr[j])
        y_pred_char.append(pr[j])
            
    if pr == tr :
        correct += 1 
    
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/valid_size))


classreport = (classification_report(y_true_char,y_pred_char))
print(classreport)
cm = (confusion_matrix(y_true_char,y_pred_char))
pl.matshow(cm)
pl.title("Confusion Matrix of Predicted Characters")
pl.colorbar()
pl.show()


mean_squared_error([ord(i) for i in y_true_char],[ord(i) for i in y_pred_char])


#Test over 300000

#train.to_csv("TrainTable.csv")
#valid.to_csv("TrainTable.csv",index=False)


valid = pd.read_csv("TrainTable.csv")
#test = pd.read_csv('HandwrittenDataset/written_name_test_v2.csv')

prediction = ["" for i in range(len(valid))]
j = 0
while(j*3000 < len(valid)):
    valid_size= 3000
    start_ind = j*3000
    valid_x = []
    print(j)
    for i in range(start_ind,min(start_ind+valid_size,len(valid))):
        #img_dir = 'HandwrittenDataset/validation_v2/validation/'+valid.loc[i, 'FILENAME']
        img_dir = 'HandwrittenDataset/train_v2/train/'+valid.loc[i, 'FILENAME']
        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        image = preprocess(image)
        image = image/255.
        valid_x.append(image)
    
    #valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)
    valid_x = np.array(valid_x).reshape(-1, 16384)
    preds = model.predict(valid_x)
    decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)[0][0])
    

    for i in range(len(valid_x)):
        prediction[start_ind + i] = (num_to_label(decoded[i]))
    j += 1 


#valid["RNNPredicted"] =  pd.Series(prediction,name = "RNNPredicted")

y_true = valid['IDENTITY']
correct_char = 0
total_char = 0
correct = 0
y_true_char = []
y_pred_char = []
for i in range(len(y_true)):
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)
    
    for j in range(min(len(tr), len(pr))):
        #if tr[-j] == pr[-j]:
        if tr[j] == pr[j]:
            correct_char += 1
        #y_true_char.append(tr[-j])
        #y_pred_char.append(pr[-j])
        y_true_char.append(tr[j])
        y_pred_char.append(pr[j])
            
    if pr == tr :
        correct += 1 
    
print("No of Images " + str(len(y_true)))
print("No of Characters over all Images " + str(len(y_true_char)))
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/len(y_true)))



print(classification_report(y_true_char,y_pred_char))

cm = (confusion_matrix(y_true_char,y_pred_char))
pl.matshow(cm)
pl.title("Confusion Matrix of Predicted Characters")
pl.colorbar()
pl.show()


from sklearn.metrics import mean_squared_error
mean_squared_error([ord(i) for i in y_true_char],[ord(i) for i in y_pred_char])


clean_result = test["IDENTITY"]
validation_df = pd.Series(prediction,name = "Predicted")
# Create 1 dataframe with both actual and OCR labels
ocr_vs_actual = pd.merge(validation_df,clean_result,right_index=True,left_index=True)

# Remove labels which do not exist
ocr_vs_actual = ocr_vs_actual.loc[ocr_vs_actual['Predicted'].notnull(), :]

# Remove spaces in OCR output
ocr_vs_actual['IDENTITY'] = ocr_vs_actual['IDENTITY'].str.replace('\\s', '', regex=True)
ocr_vs_actual.head(10)



# Create jaro-winkler similarity score
vectorized_jaro_winkler = np.vectorize(jaro_winkler)

ocr_vs_actual['SIMILARITY_SCORE'] = vectorized_jaro_winkler(ocr_vs_actual['Predicted'].str.upper(), np.where(ocr_vs_actual['IDENTITY'].isnull(), '',  ocr_vs_actual['IDENTITY'].str.upper()))

ocr_vs_actual.head(10)



# Plot histogram of similarity scores to see how well we did
plt.style.use('seaborn-white')
plt.figure(figsize=(8,3), dpi=120)
plt.hist(ocr_vs_actual['SIMILARITY_SCORE'], bins=50, alpha=0.5, color='steelblue', edgecolor='none')
plt.title('Histogram of Jaro-Winkler similarity score between label and OCR-results')
plt.xlabel("Similarity")
plt.ylabel("No of Images")
plt.show()



