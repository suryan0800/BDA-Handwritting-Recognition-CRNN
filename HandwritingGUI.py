# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:05:51 2020

@author: surya
"""

# img_viewer.py


import PySimpleGUI as sg

import os

from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
import cv2
import numpy as np 


os.chdir('C:\\Users\\surya\\Documents\\Lab\\Python\\BigDataProject')
model = tf.keras.models.load_model("tfmodelforhandwriting2")
#model = tf.keras.models.load_model("CNNModel")


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

# First the window layout in 2 columns


file_list_column = [

    [

        sg.Text("Image Folder"),

        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),

        sg.FolderBrowse(),

    ],

    [

        sg.Listbox(

            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"

        )

    ],

]

# For now will only show the name of the file that was chosen

image_viewer_column = [

    [sg.Text("Choose an image from list on left:")],

    [sg.Text(size=(40, 1), key="-TOUT-")],

    [sg.Image(key="-IMAGE-")],

]


# ----- Full layout -----

layout = [

    [

        sg.Column(file_list_column),

        sg.VSeperator(),

        sg.Column(image_viewer_column),

    ]

]

window = sg.Window("Handwritten Image Recognizer", layout)


while True:

    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:

        break
    
    
    # Folder name was filled in, make a list of files in the folder
    
    if event == "-FOLDER-":
    
        folder = values["-FOLDER-"]
        #print("hello")
    
        try:
    
            # Get list of files in folder
    
            file_list = os.listdir(folder)
    
        except:
    
            file_list = []
    
    
        fnames = [
    
            f
    
            for f in file_list
    
            if os.path.isfile(os.path.join(folder, f))
    
            and f.lower().endswith((".png", ".gif",".jpg"))
    
        ]
    
        window["-FILE LIST-"].update(fnames)
        
        
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
    
        try:
    
            filename = os.path.join(
    
                values["-FOLDER-"], values["-FILE LIST-"][0]
    
            )
            print("Hello")
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            image = preprocess(image)
            image = image/255.
            print("preprocessed")
            pred = model.predict(image.reshape(1, 256, 64, 1))
            print("predicted")
            decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], greedy=True)[0][0])
            labell = num_to_label(decoded[0])
            print("Label predicted",labell)
            print("Decoded")
            print("END")
            window["-TOUT-"].update("Predicted: " + labell)
            if filename.endswith(".jpg"):
                Image.open(filename).convert("RGB").save("ShowImage.png")
                filename = "ShowImage.png"
            
            window["-IMAGE-"].update(filename=filename)
    
        except:
    
            pass
    
window.close()