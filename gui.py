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
        self.create_widgets()
        
        # Initialize tobii_controller with configurations
        self.controller = tobii_controller(self.screen_width, self.screen_height)
        self.controller.show_status()
         
        try:
            os.makedirs(self.cal_path)
        except Exception:
            # directory already exists
            pass
                
    def create_widgets(self):        
        self.canvas = tk.Canvas(self)
        
        config_fields = ["Age (Months)", "Sex", "Severity (1-5)", "Screen size (inches)", "Distance to screen (cm)"]
        config_values = ["12", "M", "1", "27", "60"]
        self.config_panel = tk.Frame(self)
        self.config_setup(self.config_panel, config_fields, config_values)
        self.config_panel.pack(side=tk.TOP, pady=(self.screen_height / 2, 0))
        
        self.calibrate_button = tk.Button(self)
        self.calibrate_button["text"] = "Make transformation"
        self.calibrate_button["fg"]   = "white"
        self.calibrate_button["bg"]   = "#FFA500"
        self.calibrate_button["command"] = self.start_calibration_exercise
        
        self.training_button = tk.Button(self)
        self.training_button["text"] = "Start training"
        self.training_button["fg"]   = "white"
        self.training_button["bg"]   = "#4CAF50"
        self.training_button["command"] = self.start_training_exercise
        
        self.psycho_button = tk.Button(self)
        self.psycho_button["text"] = "Custom calibration"
        self.psycho_button["fg"]   = "white"
        self.psycho_button["bg"]   = "#2196F3"
        self.psycho_button["command"] = self.psycho_start
        
        self.shutdown_button = tk.Button(self)
        self.shutdown_button["text"] = "Shutdown"
        self.shutdown_button["fg"]   = "white"
        self.shutdown_button["bg"]   = "#f44336"
        self.shutdown_button["command"] = self.client_exit
        
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
            
            self.config_dict = {}
            for f, e in zip(field_names, entry_texts):
                self.config_dict[f] = e
            
            config_writer.writerow(self.config_dict)
                    
        print("Configurations saved")
        self.config_panel.pack_forget()
        self.hide_canvas() 
        
        
        self.controller.set_dist_to_screen(self.config_dict["Distance to screen (cm)"])
    
    def hide_canvas(self):
        self.calibrate_button.pack(side=tk.TOP, pady=(self.screen_height / 2 - 50, 10))
        self.training_button.pack(side=tk.TOP, pady=(0, 10))
        self.psycho_button.pack(side=tk.TOP, pady=(0, 10))
        self.shutdown_button.pack(side=tk.TOP)
        
        
        self.canvas.pack_forget()
        
    def show_canvas(self):
        self.shutdown_button.pack_forget()
        self.calibrate_button.pack_forget()
        self.training_button.pack_forget()
        self.psycho_button.pack_forget()
        
        self.canvas.pack(fill="both", expand=True)
        

    def start_training_exercise(self):
        
        # Start eye traking
        if self.controller.eyetracker != None:
            print("Starting simulation with eye tracker")
#            self.eye_tracking.start_gaze_tracking()
        else:
            print("Starting simulation without eye tracker")
        
        

    def stop_training_exercise(self):
        print("Simulation ended")
        
        # Stop eye tracking
        if self.controller.eyetracker != None:
            pass
#            self.eye_tracking.end_gaze_tracking()
#            
#            self.training_file_index = self.training_file_index + 1
#            
#            training_filename = self.session_path + "training_with_cal_" + str(self.cal_file_index) + "/training_" + str(self.training_file_index) + ".csv"
#            
#            # PYTHON 2.x
#            with open(training_filename, mode='wb') as gaze_data_file:
#            
#                field_names = [data for data in self.eye_tracking.gaze_params]
#                gaze_data_writer = csv.DictWriter(gaze_data_file, fieldnames=field_names, delimiter=";")
#            
#                gaze_data_writer.writeheader()
#                for gaze_data in self.eye_tracking.global_gaze_data:
#                    gaze_data_writer.writerow(gaze_data)
#                    
#            try:
#                self.analyzer.analyze(training_filename)
#            except:
#                print("Bad data obtained")
       
        # Hide canvas
        # Show button after exercise
        self.hide_canvas()
        
    def start_calibration_exercise(self):
        self.controller.make_transformation()
        
    def stop_calibration_exercise(self):
        print("Simulation ended")
        
        # Stop eye tracking
        if self.controller.eyetracker != None:
#            self.eye_tracking.end_gaze_tracking()
#            
#            self.cal_file_index = self.cal_file_index + 1
#            self.training_file_index = 0
#            
#            try:
#                os.makedirs(self.session_path + "training_with_cal_" + str(self.cal_file_index) + "/")
#            except Exception:
#                # directory already exists
#                pass
#            
#            cal_filename = self.cal_path + "cal_" + str(self.cal_file_index) + ".csv"
#            
#            # PYTHON 2.x
#            with open(cal_filename, mode='wb') as gaze_data_file:
#            
#                field_names = [data for data in self.eye_tracking.gaze_params]
#                gaze_data_writer = csv.DictWriter(gaze_data_file, fieldnames=field_names, delimiter=";")
#            
#                gaze_data_writer.writeheader()
#                for gaze_data in self.eye_tracking.global_gaze_data:
#                    gaze_data_writer.writerow(gaze_data)
#
#                    
#            try:
#                self.analyzer.setup(self.config_filename, cal_filename)
#                self.analyzer.analyze(cal_filename)
#            except:
#                print("Bad data obtained")
                
            pass
   
        # Hide canvas
        # Show button after exercise
        self.hide_canvas()
       
   
        
    def psycho_start(self):
        
        # we can only check if there is an existing eye tracking device.
        # this will still fail whenever the device is there but turned off 
        # TobiiProSDK does not support activity checks this for python it seems..
        if self.controller.eyetracker != None:
            self.controller.start_gaze_trace()
           
            
            
    def client_exit(self):
        print("Shutting down")
        self.quit()

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
