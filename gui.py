#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:41:26 2019

@author: sebastiannyholm
"""

try:
    # for Python2
    import Tkinter as tk
except ImportError:
    # for Python3
    import tkinter as tk

from PIL import Image, ImageTk
from eye_tracking import EyeTracking
import config
 
class Application(tk.Frame):
    
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        
        # Get width and height of screen
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        
        self.pack(fill="both", expand=True)
        self.create_widgets()
        
        try:
            self.eye_tracking = EyeTracking()
        except Exception as eye_tracker_exception:
            print(eye_tracker_exception)

    def create_widgets(self):
        
        
        '''
        self.shapes = []
        self.shapes.append(Shape("Alien"))
        self.shapes.append(Shape("Cat"))
        self.shapes[0].add_image("alien.jpg", 250, 250)
        self.shapes[0].set_animation(9, 9, True)
        self.shapes[0].create_image(self.canvas, tk.NW)

        self.shapes[1].add_image("cat.jpg", 250, 250)
        self.shapes[1].set_animation(12, 17, True)
        self.shapes[1].create_image(self.canvas, tk.NW)

        #self.shape = self.canvas.create_image(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, image=photo, anchor = NW)
        #self.canvas.image = photo
        
        #self.alien.speedx = 9
        #self.alien.speedy = 9
        #self.alien.active = True
        for shape in self.shapes:
            self.move(shape)
        '''
        '''
        # creating a menu instance
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        # create the file object)
        file = tk.Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        #added "file" to our menu
        menu.add_cascade(label="File", menu=file)


        # create the file object)
        edit = tk.Menu(menu)
        edit.add_command(label="Start", command=self.start_exercise)
        edit.add_command(label="Stop", command=self.stop_exercise)
        '''
        '''
        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        for shape in self.shapes:
            edit.add_command(label="Start " + shape.name, command=lambda shape=shape: self.start_animation(shape))
            edit.add_command(label="Stop " + shape.name, command=lambda shape=shape: self.stop_animation(shape))
        '''
        '''
        #added "file" to our menu
        menu.add_cascade(label="Edit", menu=edit)
        '''
        
        self.canvas = tk.Canvas(self)
        
        self.ball1 = Ball(self.canvas, 9, 9, 110, 110)
        self.ball2 = Ball(self.canvas, self.screen_width - 10, self.screen_height - 10, self.screen_width - 110, self.screen_height - 110)
        
        self.start_button = tk.Button(self)
        self.start_button["text"] = "Start exercise"
        self.start_button["command"] = self.start_exercise
        
        self.shutdown_button = tk.Button(self)
        self.shutdown_button["text"] = "Shutdown"
        self.shutdown_button["command"] = self.client_exit
            
        self.hide_exercise()
        
    def hide_exercise(self):
        self.start_button.pack(side=tk.TOP, pady=(self.screen_height / 2 - 50, 10))
        self.shutdown_button.pack(side=tk.TOP)
        
        self.canvas.pack_forget()
        
    def show_exercise(self):
        self.shutdown_button.pack_forget()
        self.start_button.pack_forget()
        
        self.canvas.pack(fill="both", expand=True)
        
        # Show balls after 5 and 10 secounds
        self.canvas.after(0, lambda ball=None: self.animate(ball, 0))
        self.canvas.after(5000, lambda ball=self.ball1: self.animate(ball, 5000))
        self.canvas.after(10000, lambda ball=self.ball2: self.animate(ball, 5000))
        self.canvas.after(15000, lambda ball=None: self.animate(ball, 0))

    def animate(self, ball, delay_before_hide):
        
        # Show ball <if not None>
        if ball != None:
            ball.show(delay_before_hide)
        
        # Set current target points for Eye Tracker
        if self.eye_tracking != None:
            if ball != None:
                
                center_x_norm = (ball.center_x - config.min_x) / (config.max_x - config.min_x)
                center_y_norm = (ball.center_y - config.min_y) / (config.max_y - config.min_y)
                
                self.eye_tracking.set_current_target(center_x_norm, center_y_norm)
            else:
                self.eye_tracking.set_current_target(0.5, 0.5)

    def start_exercise(self):
        print("Exercise started")
        
        # Start eye traking
        self.eye_tracking.start_gaze_tracking()
        
        # Hide button
        # Show canvas
        # Show elements
        self.show_exercise()
        
        # End after 20 seconds
        self.canvas.after(20000, self.stop_exercise)
        
    def stop_exercise(self):
        print("Exercise stopped")
        
        # Stop eye traking
        self.eye_tracking.end_gaze_tracking()
       
        # Hide canvas
        # Show button after exercise
        self.hide_exercise()
        
        
    def start_animation(self, shape):
        if not shape.active:
            shape.active = True        
            self.move(shape)
            
    def stop_animation(self, shape):
        shape.active = False
    
    def move(self, shape):
        if shape.active:
            
            pos = self.canvas.coords(shape.object)
            
            if pos[0] < 0 or self.screen_width < pos[0] + shape.width:
                shape.speedx *= -1
            if pos[1] < 0 or self.screen_width < pos[1] + shape.height:
                shape.speedy *= -1
    
            self.canvas.move(shape.object, shape.speedx, shape.speedy)
    
            self.canvas.after(50, lambda: self.move(shape))
        
    def client_exit(self):
        print("Shutting down")
        self.quit()


class Ball:
    def __init__(self, canvas, left_upper_x, left_upper_y, right_bottom_x, right_bottom_y):
        self.left_upper_x = left_upper_x
        self.left_upper_y = left_upper_y
        self.right_bottom_x = right_bottom_x
        self.right_bottom_y = right_bottom_y
        self.canvas = canvas

        self.center_x = self.left_upper_x + float(self.right_bottom_x - self.left_upper_x) / 2
        self.center_y = self.left_upper_y + float(self.right_bottom_y - self.left_upper_y) / 2
        
    def show(self, delay_before_hide):
        self.ball = self.canvas.create_oval(self.left_upper_x, self.left_upper_y, self.right_bottom_x, self.right_bottom_y, fill="red")
        self.canvas.after(delay_before_hide, self.hide)
        
    def hide(self):
        self.canvas.delete(self.ball)

class Shape:
    
    def __init__(self, name):
        self.name = name
    
    def add_image(self, image, width, height):
        image = Image.open("images/"+image)
        image = image.resize((width, height), Image.ANTIALIAS) #The (250, 250) is (height, width)
        self.image = ImageTk.PhotoImage(image)
        self.width = width;
        self.height = height;

    def create_image(self, canvas, anchor):
        self.object = canvas.create_image(self.width, self.height, image=self.image, anchor = anchor)
    
    def set_animation(self, speedx, speedy, active):
        self.speedx = speedx
        self.speedy = speedy
        self.active = active

root = tk.Tk()
#root = tk.Toplevel()

# use the next line if you also want to get rid of the titlebar
root.attributes("-fullscreen", True)

# Set size of window to screen size
#root.geometry("%dx%d+0+0" % (w, h))
#root.geometry("1200x900")

app = Application(master=root)
app.mainloop()
root.destroy()