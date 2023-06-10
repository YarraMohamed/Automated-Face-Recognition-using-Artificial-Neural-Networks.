# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:06:00 2022

@author: 20100
"""

from tkinter import*
import tkinter as tk
# from tkinter import tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import cv2

class window:
    def __init__(self,master):
        master.configure(bg="dark cyan")
        master.title("Browse a file")
        self.label1=Label(text="Hello",
        fg="white",
        font="bold",
        bg="black",
        width=20,
        height=4).pack()
        self.spacer1=Label(text="",bg="dark cyan",height=3).pack()
        self.label2=Label(text="Please choose a file",
        fg="white",
        font="bold",
        bg="black",
        width=50,
        height=4).pack()
        self.spacer2=Label(text="",bg="dark cyan",height=1).pack()
        self.filename=filedialog.askopenfilename(title="Choose Image",initialdir="C:\\Users\\Lenovo\\Desktop\\AI Project\\105_classes_pins_dataset",filetypes=[('Jpg Files', '*.jpg')])
        self.value=tk.StringVar()
        self.out_label=self.value.get()
        
        
        self.browse_Button=Button(text="View prediction",
        font="bold",
        width=30,
        height=3,
        bg="medium spring green",
        fg="black",
        command =lambda:upload_file(self)).pack()
        self.spacer3=Label(text="",bg="dark cyan",height=1).pack()
        def upload_file(self):
            global img
            img_size=50
            self.value=tk.Label(textvariable=self.value)
            my_label=Label(text=self.out_label)
            my_label.pack()
            Spacer4=Label(text="",bg="dark cyan",height=1).pack()
            img = ImageTk.PhotoImage(file=self.filename)
            label_img=Label(image=img).pack()
          