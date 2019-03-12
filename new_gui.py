# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:24:14 2019

@author: s144451, s144434
"""



try:
    # for Python2
    import Tkinter as tk
except ImportError:
    # for Python3
    import tkinter as tk

from psychopy_tobii_controller.tobii_wrapper import tobii_controller
import datetime
import csv
import gaze_data_analyzer as gda
import os


class Application(tk.Frame):
    
    analyzer = gda.GazeDataAnalyzer()
    
    session_path = "session_data/" + datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S/")
    
    cal_file_index = 0
    training_file_index = 0
    
    config_filename = session_path + "config.csv"
    
    cal_path = session_path + "calibrations/"
    
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        
        # Get width and height of screen
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        
        self.pack(fill="both", expand=True)
        
        
        # Initialize tobii_controller with configurations 
        self.controller = tobii_controller(self.screen_width, self.screen_height)
        self.controller.show_status()
        
        try:
            os.makedirs(self.cal_path)
        except Exception:
            # directory already exists
            pass
        
        config_fields = ["Age (Months)", "Sex", "Severity (1-5)", "Screen size (inches)", "Distance to screen (cm)"]
        config_values = ["12", "M", "1", "27", "60"]
        self.panel_config = tk.Frame(self)
        self.config_setup(self.panel_config, config_fields, config_values)
        self.panel_config.pack(side=tk.TOP, pady=(self.screen_height / 2, 0))
        
        
    def show_main_panel(self):

        self.btn_test = tk.Button(self)
        self.btn_test["text"] = "TEST"
        self.btn_test["fg"]   = "white"
        self.btn_test["bg"]   = "#000000"
        self.btn_test["command"] = self.test
        self.btn_test.pack(side=tk.TOP, pady=(0, 10))
        
        
        
        
        
        self.btn_default_test = self.make_test_button("Test default", "default")
        self.btn_2p_test = self.make_test_button("Test 2-point", "2p")
        self.btn_5p_test = self.make_test_button("Test 5-point", "5p")
        self.btn_5p_img_test = self.make_test_button("Test 5-point (image)", "5p_img")   
        
        self.btn_show_status = tk.Button(self)
        self.btn_show_status["text"] = "Check eye position"
        self.btn_show_status["fg"]   = "white"
        self.btn_show_status["bg"]   = "#2196F3"
        self.btn_show_status["command"] = self.show_status
    
        self.btn_shutdown = tk.Button(self)
        self.btn_shutdown["text"] = "Shutdown"
        self.btn_shutdown["fg"]   = "white"
        self.btn_shutdown["bg"]   = "#f44336"
        self.btn_shutdown["command"] = self.client_exit
        
        # pack it all
        self.btn_default_test.pack(side=tk.TOP, pady=(self.screen_height / 2 - 50, 10))
        self.btn_2p_test.pack(side=tk.TOP, pady=(0, 10))
        self.btn_5p_test.pack(side=tk.TOP, pady=(0, 10))
        self.btn_5p_img_test.pack(side=tk.TOP, pady=(0, 10))
        self.btn_show_status.pack(side=tk.TOP, pady=(0, 10))
        self.btn_shutdown.pack(side=tk.TOP)
        
        
    def test(self):
        self.custom_calibration(5, "img")
        
        
    def make_test_button(self, title, cal_type):
        btn = tk.Button(self)
        btn["text"] = title
        btn["fg"]   = "white"
        btn["bg"]   = "#4CAF50"
        btn["command"] = lambda t=cal_type: self.run_test(t)
        return btn
        
    def run_test(self, cal_type):
        if cal_type == "default":
            pass
        elif cal_type == "2p  ":
            self.custom_calibration(2)
        elif cal_type == "5p":
            self.custom_calibration(5)
        elif cal_type == "5p_img":
            self.custom_calibration(5, "img")
        
        self.make_transformation()
        self.training_exercise("fixation")
        self.training_exercise("pursuit", "linear")
        self.training_exercise("pursuit", "spiral")
        
        
    def config_setup(self, root, fields, values):
        entries = []
        for field, value in zip(fields, values):
            row = tk.Frame(root)
            label = tk.Label(row, width=15, text=field, anchor='w')
            entry = tk.Entry(row)
            entry.insert(tk.END, value)
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            label.pack(side=tk.LEFT)
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            entries.append((field, entry))
        btn_config_save = tk.Button(root)
        btn_config_save["text"] = "Save configuration"
        btn_config_save["fg"]   = "black"
        btn_config_save["bg"]   = "#b2b2b2"
        btn_config_save["command"] = lambda e=entries: self.config_save(e)
        btn_config_save.pack(side=tk.LEFT, padx=5, pady=10)
    
    def config_save(self, entries):
        
        # PYTHON 2.x
        with open(self.config_filename, mode='wb') as csv_file:
            field_names = [row[0] for row in entries]
            entry_texts = [row[1].get() for row in entries]
            
            field_names.extend(["Screen width (px)", "Screen height (px)"])
            entry_texts.extend([self.screen_width, self.screen_height])
            
            config_writer = csv.DictWriter(csv_file, fieldnames=field_names, delimiter=";")
            config_writer.writeheader()
            
            config_dict = {}
            for f, e in zip(field_names, entry_texts):
                config_dict[f] = e
            
            config_writer.writerow(config_dict)
            
            self.controller.set_dist_to_screen(config_dict["Distance to screen (cm)"])        
            
        print("Configurations saved")
        self.panel_config.pack_forget()
        self.show_main_panel() 
    
    

    def show_status(self):
        self.controller.show_status()
        
    def training_exercise(self, training_type="fixation", pathing_type="linear"):
        # Start eye traking
        if training_type == "fixation":
            self.controller.start_fixation_exercise()
            
        elif training_type == "pursuit":
            self.controller.start_pursuit_exercise(pathing=pathing_type)
            
        
    def make_transformation(self):
        self.controller.make_transformation()
   
        
    def custom_calibration(self, num_points, stim_type="default"):
        
        # we can only check if there is an existing eye tracking device.
        # this will still fail whenever the device is there but turned off 
        # TobiiProSDK does not support activity checks this for python it seems..
        self.controller.start_custom_calibration(num_points, stim_type=stim_type)
            
    def client_exit(self):
        print("Shutting down")
        self.quit()

root = tk.Tk()
#root = tk.Toplevel()

# use the next line if you also want to get rid of the titlebar
root.attributes("-fullscreen", True)

# set focus
root.lift()
root.attributes("-topmost", True)

# Set size of window to screen size
#root.geometry("%dx%d+0+0" % (w, h))
#root.geometry("1200x900")

app = Application(master=root)
app.mainloop()
root.destroy()
