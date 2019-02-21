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
import datetime
import csv
 
class Application(tk.Frame):
    
    
    config_path = "session_data/config_files/"
    gaze_data_path = "session_data/gaze_data/"
    session_file = datetime.datetime.now().strftime("%A, %d. %B %Y %I.%M.%S %p")+".csv"
    
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        
        # Get width and height of screen
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        
        self.pack(fill="both", expand=True)
        self.create_widgets()
        
        self.eye_tracking = None
        try:
            self.eye_tracking = EyeTracking()
        except Exception as eye_tracker_exception:
            print(eye_tracker_exception)

    def create_widgets(self):        
        self.canvas = tk.Canvas(self)
        
        self.ball1 = Ball(self.canvas, 10, 10, 110, 110)
        self.ball2 = Ball(self.canvas, self.screen_width - 10, self.screen_height - 10, self.screen_width - 110, self.screen_height - 110)
        
        config_fields = ["Age (Months)", "Sex", "Severity (1-5)"]
        self.config_panel = tk.Frame(self)        
        self.config_setup(self.config_panel, config_fields)
        self.config_panel.pack(side=tk.TOP, pady=(self.screen_height / 2, 0))
        
        self.start_button = tk.Button(self)
        self.start_button["text"] = "Start exercise"
        self.start_button["fg"]   = "white"
        self.start_button["bg"]   = "#4CAF50"
        self.start_button["command"] = self.start_exercise
        
        self.shutdown_button = tk.Button(self)
        self.shutdown_button["text"] = "Shutdown"
        self.shutdown_button["fg"]   = "white"
        self.shutdown_button["bg"]   = "#f44336"
        self.shutdown_button["command"] = self.client_exit
        
    def config_setup(self, root, fields):
        entries = []
        for field in fields:
            row = tk.Frame(root)
            label = tk.Label(row, width=15, text=field, anchor='w')
            entry = tk.Entry(row)
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            label.pack(side=tk.LEFT)
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            entries.append((field, entry))
        btn_config_save = tk.Button(root)
        btn_config_save["text"] = "Save configuration"
        btn_config_save["fg"]   = "black"
        btn_config_save["bg"]   = "#b2b2b2"
        btn_config_save["command"] = lambda e=entries: self.config_save(e)
        btn_config_save.pack(side=tk.LEFT, padx=15, pady=10)
    
    def config_save(self, entries):
        filename = self.config_path + self.session_file
        # PYTHON 2.x
        with open(filename, mode='wb') as csv_file:
            config_writer = csv.writer(csv_file)
            field_names = [row[0] for row in entries]
            entry_texts = [row[1].get() for row in entries]
            
            field_names.extend(["Screen width", "Screen height"])
            entry_texts.extend([self.screen_width, self.screen_height])
            
            config_writer.writerow(field_names)     # field/label names
            config_writer.writerow(entry_texts)     # text of entries
            
        print("Configurations saved")
        self.config_panel.pack_forget()
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
        self.ball1.animate(5000,5000)
        self.ball2.animate(10000,5000)
        

    def start_exercise(self):
        
        # Start eye traking
        if self.eye_tracking != None:
            print("Starting simulation with eye tracker")
            self.eye_tracking.start_gaze_tracking()
        else:
            print("Starting simulation without eye tracker")
        
        # Hide button
        # Show canvas
        # Show elements
        self.show_exercise()
        
        # End after 20 seconds
        self.canvas.after(20000, self.stop_exercise)
        
    def stop_exercise(self):
        print("Simulation ended")
        
        # Stop eye traking
        if self.eye_tracking != None:
            self.eye_tracking.end_gaze_tracking()
            
            filename = self.gaze_data_path + self.session_file
            # PYTHON 2.x
            with open(filename, mode='wb') as gaze_data_file:
            
                field_names = [data for data in self.eye_tracking.gaze_params]
                gaze_data_writer = csv.DictWriter(gaze_data_file, fieldnames=field_names, delimiter=";")
            
                gaze_data_writer.writeheader()
                for gaze_data in self.eye_tracking.global_gaze_data:
                    gaze_data_writer.writerow(gaze_data)
                    
       
       
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
    def __init__(self, canvas, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.canvas = canvas

    def animate(self, delay_show, delay_hide):
        self.canvas.after(delay_show, self.show)
        self.canvas.after(delay_show + delay_hide,self.hide)
        
    def show(self):
        self.ball = self.canvas.create_oval(self.x1, self.y1, self.x2, self.y2, fill="red")
        
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