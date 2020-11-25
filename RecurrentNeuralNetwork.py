# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:26:26 2020

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
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from Levenshtein import jaro_winkler
from IPython.display import display
from PIL import Image
import pylab as pl

os.chdir('C:\\Users\\surya\\Documents\\Lab\\Python\\BigDataProject')

train = pd.read_csv("TrainTable.csv")
valid = pd.read_csv("ValidTable.csv")

#train.to_csv("TrainTable.csv")
#valid.to_csv("ValidTable.csv")

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


start_ind_train = 0
train_size = 30000
valid_size= 3000
start_ind_valid = 0


#pca = PCA(n_components=4096)

#train_x = pca.fit_transform(train_x)
#np.save("train_x.npy",train_x)
#valid_x = pca.fit_transform(valid_x)
#np.save("valid_x.npy",valid_x)

#train_x = np.load("train_x.npy")
#valid_x = np.load("valid_x.npy")

#np.save("ProcessedImageTrain_x.npy",train_x)
#np.save("ProcessedImageValid_x.npy",valid_x)

train_x = np.load("ProcessedImageTrain_x.npy")
valid_x = np.load("ProcessedImageValid_x.npy")


alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 # max length of input labels
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


train_y = np.ones([train_size, max_str_len]) * -1
train_label_len = np.zeros([train_size, 1])
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps-2)
train_output = np.zeros([train_size])

for i in range(start_ind_train, start_ind_train + train_size):
    train_label_len[i - start_ind_train] = len(train.loc[i, 'IDENTITY'])
    train_y[i - start_ind_train, 0:len(train.loc[i, 'IDENTITY'])]= label_to_num(train.loc[i, 'IDENTITY'])    



valid_y = np.ones([valid_size, max_str_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])

for i in range(start_ind_valid ,start_ind_valid + valid_size):
    valid_label_len[i - start_ind_valid] = len(valid.loc[i, 'IDENTITY'])
    valid_y[i - start_ind_valid, 0:len(valid.loc[i, 'IDENTITY'])]= label_to_num(valid.loc[i, 'IDENTITY'])    


#https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53


input_data = Input(shape=(16384))

inner = Reshape(target_shape=((64,256)))(input_data)
inner = Dense(64, activation='relu', kernel_initializer='he_normal')(inner)
#inner = Dense(4096, activation='relu', kernel_initializer='he_normal')(inner)
#inner = Reshape(target_shape=((64,64)))(input_data)
## RNN
#https://colah.github.io/posts/2015-08-Understanding-LSTMs/
inner = (LSTM(256, return_sequences=True))(inner)
inner = (LSTM(256, return_sequences=True))(inner)
inner = (LSTM(256, return_sequences=True))(inner)
## OUTPUT
inner = Dense(num_of_characters, kernel_initializer='he_normal')(inner)
y_pred = Activation('softmax')(inner)

model = Model(inputs=input_data, outputs=y_pred)
model.summary()


#https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c
# the ctc loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)




labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)


# the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr = 0.001))

history = model_final.fit(x=[train_x, train_y, train_input_len, train_label_len], y=train_output,  validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output), epochs=30, batch_size=256)


print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#valid_x = valid_x.reshape(3000,-1)
preds = model.predict(valid_x)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)[0][0])

prediction = []
for i in range(valid_size):
    prediction.append(num_to_label(decoded[i]))



#model.save("RNNModel5")

#model = tf.keras.models.load_model("RNNModel4")




y_true = valid.loc[0:valid_size-1, 'IDENTITY']
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
        if tr[-j] == pr[-j]:
            correct_char += 1
        y_true_char.append(tr[-j])
        y_pred_char.append(pr[-j])
            
    if pr == tr :
        correct += 1 
        
print("No of Images " + str(len(y_true)))
print("No of Characters over all Images " + str(len(y_true_char)))
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/valid_size))




#test = pd.read_csv("HandwrittenDataset/written_name_train_v2.csv")
test = pd.read_csv('HandwrittenDataset/written_name_test_v2.csv')

plt.figure(figsize=(15, 10))
start_img_ind = 1000
for i in range(start_img_ind,start_img_ind+12):
    ax = plt.subplot(4, 3, i+1-start_img_ind)
    img_dir = 'HandwrittenDataset/test_v2/test/'+test.loc[i, 'FILENAME']
    #img_dir = 'HandwrittenDataset/train_v2/train/'+test.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    
    image = preprocess(image)
    image = image/255.
    pred = model.predict(image.reshape(1,-1))
    #pred = model.predict(image.reshape(1,256,64,1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], greedy=True)[0][0])
    plt.title(num_to_label(decoded[0]), fontsize=12)
    
    plt.axis('off')
    
plt.subplots_adjust(wspace=0.2, hspace=-0.8)


print(classification_report(y_true_char,y_pred_char))

cm = (confusion_matrix(y_true_char,y_pred_char))
pl.matshow(cm)
pl.title("Confusion Matrix of Predicted Characters")
pl.colorbar()
pl.show()




clean_result = y_true
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
plt.show()





















