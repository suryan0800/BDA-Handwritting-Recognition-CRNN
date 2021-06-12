# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 18:18:15 2020

@author: surya
"""

from tkinter import *
from tkinter.colorchooser import askcolor
import tkinter as tk
import numpy as np
import cv2
from PIL import Image,ImageGrab, EpsImagePlugin
import os
from IPython.display import display

import tensorflow as tf
from tensorflow.keras import backend as K

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        
        
        self.model = tf.keras.models.load_model("tfmodelforhandwriting2")
        #self.model = tf.keras.models.load_model("CNNModel")
        self.alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
        
        self.root = Tk()
        self.root.title("Handwriting Virtual Keyboard")
        
        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)

        self.undo_button = Button(self.root, text='undo', command=self.undo)
        self.undo_button.grid(row=0, column=3)
        
        self.predict_button = Button(self.root, text='Predict', command=self.predict)
        self.predict_button.grid(row=0, column=3)
        
        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)
        self.choose_size_button.set(2)

        self.c = Canvas(self.root, bg='white', width=256, height=64)
        self.c.grid(row=1, columnspan=5)

        self.l = Label(self.root, text = "Predicted: ") 
        self.l.config(font =("Courier", 10)) 
        self.l.grid(row=2,column=0,columnspan=4)
        
        self.T = Text(self.root, height = 1, width = 20) 
        self.T.grid(row=2,column = 4,columnspan=4)
        self.T.config(font=12)
        
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.root.bind('z',self.undo)
        #self.root.bind('s',self.predict)
        self.stack = []

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)
        
    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        
        if self.eraser_on:
            paint_color = 'white' 
            self.line_width = 20
        else:
            paint_color = self.color
            self.line_width = self.choose_size_button.get()
            
        if self.old_x and self.old_y:
            x = self.c.create_line(self.old_x, self.old_y, event.x, event.y, width=self.line_width, fill=paint_color, capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.stack.append(x)
        self.old_x = event.x
        self.old_y = event.y

    def undo(self,event=None):
        if len(self.stack) > 0:
            x = self.stack.pop() 
            self.c.delete(x) 
            #print(np.asarray(self.c))
    def reset(self, event):
        self.old_x, self.old_y = None, None

    def predict(self,event=None):
        self.c.postscript(file = 'sample' + '.eps') 
        # use PIL to convert to PNG 
        print("Hello")
        img = Image.open('sample' + '.eps').save("sample2.png")
        filename="sample2.png"
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = self.preprocess(image)
        image = image/255.
        print("preprocessed")
        pred = self.model.predict(image.reshape(1, 256, 64, 1))
        print("predicted")
        decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], greedy=True)[0][0])
        labell = self.num_to_label(decoded[0])
        print("Label predicted",labell)
        print("Decoded")
        print("END")
    
        self.T.delete("1.0","end")
        self.T.insert("end",labell)
        




    def preprocess(self,img):
        
        
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





    def label_to_num(self,label):
        label_num = []
        for ch in label:
            label_num.append(self.alphabets.find(ch))
            
        return np.array(label_num)

    def num_to_label(self,num):
        ret = ""
        for ch in num:
            if ch == -1:  # CTC Blank
                break
            else:
                ret+=self.alphabets[ch]
        return ret

if __name__ == '__main__':
    #os.chdir('C:\\Users\\surya\\Documents')
    os.chdir('C:\\Users\\surya\\Documents\\Lab\\Python\\BigDataProject')
    EpsImagePlugin.gs_windows_binary =  r'C:\Program Files\gs\gs9.53.3\bin\gswin64c.exe'
    Paint()
    
