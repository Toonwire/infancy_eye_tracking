# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:40:39 2019

@author: Toonw
"""


import tkinter as tk
from PIL import Image, ImageTk


class Application(tk.Frame):
    CANVAS_WIDTH = 800
    CANVAS_HEIGHT = 500
    
    IMAGE_WIDTH = 100
    IMAGE_HEIGHT = 100

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.QUIT = tk.Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit
        self.QUIT.pack({"side": "bottom"})
        
        self.BTN_MOVE_IMG = tk.Button(self)
        self.BTN_MOVE_IMG["text"] = "MOVE dog"
        self.BTN_MOVE_IMG["fg"]   = "blue"
        self.BTN_MOVE_IMG["command"] = lambda: self.img_move(0)
        self.BTN_MOVE_IMG.pack({"side": "bottom"})
        
        self.BTN_MOVE_IMG = tk.Button(self)
        self.BTN_MOVE_IMG["text"] = "MOVE dog2"
        self.BTN_MOVE_IMG["fg"]   = "blue"
        self.BTN_MOVE_IMG["command"] = lambda: self.img_move(1)
        self.BTN_MOVE_IMG.pack({"side": "bottom"})
        
        self.canvas = tk.Canvas(self, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg="grey")
        self.canvas.images = []
        self.canvas.pack()
        
        img = Image.open("alien.jpg")
        img = img.resize((self.IMAGE_WIDTH,self.IMAGE_HEIGHT), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        dog = CanvasImage(photo, shape=self.canvas.create_image(self.CANVAS_WIDTH/2,self.CANVAS_HEIGHT/2, anchor=tk.NW, image=photo))
        self.canvas.images.append(dog)
        
        img = Image.open("cat.jpg")
        img = img.resize((self.IMAGE_WIDTH,self.IMAGE_HEIGHT), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        dog2 = CanvasImage(photo, shape=self.canvas.create_image(self.CANVAS_WIDTH/4,self.CANVAS_HEIGHT/4, anchor=tk.NW, image=photo))
        self.canvas.images.append(dog2)
        

    def img_move(self, canvas_image_id):
        self.canvas.images[canvas_image_id].is_moving = True if self.canvas.images[canvas_image_id].is_moving == False else False
        self.move(canvas_image_id)
        
    def move(self, canvas_image_id):
        if self.canvas.images[canvas_image_id].is_moving:
            self.img_update(canvas_image_id)
            self.master.after(40, lambda: self.move(canvas_image_id)) # changed from 10ms to 30ms
            
    def img_update(self, canvas_image_id):
        self.canvas.move(self.canvas.images[canvas_image_id].shape, self.canvas.images[canvas_image_id].speedx, self.canvas.images[canvas_image_id].speedy)
        pos = self.canvas.coords(self.canvas.images[canvas_image_id].shape)
        if pos[0] + self.IMAGE_WIDTH >= self.CANVAS_WIDTH or pos[0] <= 0:
            self.canvas.images[canvas_image_id].speedx *= -1
        if pos[1] + self.IMAGE_HEIGHT >= self.CANVAS_HEIGHT or pos[1] <= 0:
            self.canvas.images[canvas_image_id].speedy *= -1
    
    
class CanvasImage():
    
    def __init__(self, img, shape=0, speedx=5, speedy=5, is_moving=False):
        self.img = img
        self.shape = shape
        self.speedx = speedx
        self.speedy = speedy
        self.is_moving = is_moving
    
    
    

GUI_WIDTH = 800
GUI_HEIGHT = 800
root = tk.Tk()
root.geometry(("%dx%d" % (GUI_WIDTH, GUI_HEIGHT)))
app = Application(master=root)
app.mainloop()
root.destroy()
    

















