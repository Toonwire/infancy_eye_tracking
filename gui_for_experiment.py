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
from multiprocessing import Process



class Application(tk.Frame):
    
    analyzer = gda.GazeDataAnalyzer()
    
    session_path = "session_data/" + datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S") + "/"
    
    config_filename = session_path + "config.csv"
    test_folder = None
    
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        
        # Get width and height of screen
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        self.screen_size_inches = 27
        
        self.pack(fill="both", expand=True)
        
        
        # Initialize tobii_controller with configurations 
        self.controller = tobii_controller(self.screen_width, self.screen_height)
        self.controller.subscribe_dict()
        
        self.status_admin_process = Process(target=self.controller.show_status_admin(screen=0))
        self.status_admin_process.start()
        
        
        
        self.controller.show_status(screen=1)
        
        try:
            os.makedirs(self.session_path)
        except Exception:
            # directory already exists
            pass
        
        self.config_setup()
        
        
             
    def config_setup(self):
        self.panel_config = tk.Frame(self)
        
        fields = ["Age (Months)", "Sex", "Distance to screen (cm)"]
        values = ["12", "M", "60"]
        
        entries = []
        for field, value in zip(fields, values):
            row = tk.Frame(self.panel_config)
            label = tk.Label(row, width=20, text=field, anchor='w')
            entry = tk.Entry(row)
            entry.insert(tk.END, value)
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            label.pack(side=tk.LEFT)
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            entries.append((field, entry))
            
        btn_config_save = tk.Button(self.panel_config)
        btn_config_save["text"] = "Save configuration"
        btn_config_save["fg"]   = "black"
        btn_config_save["bg"]   = "#b2b2b2"
        btn_config_save["command"] = lambda e=entries: self.config_save(e)
        btn_config_save.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.panel_config.pack(side=tk.TOP, pady=(10,10))
    
    def config_save(self, entries):
        
        # PYTHON 2.x
        with open(self.config_filename, mode='wb') as csv_file:
            field_names = [row[0] for row in entries]
            entry_texts = [row[1].get() for row in entries]
            
            field_names.extend(["Screen size (inches)", "Screen width (px)", "Screen height (px)"])
            entry_texts.extend([self.screen_size_inches, self.screen_width, self.screen_height])
            
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
        
    
    

        
    def show_main_panel(self):

        self.btn_test = tk.Button(self)
        self.btn_test["text"] = "TEST"
        self.btn_test["fg"]   = "white"
        self.btn_test["bg"]   = "#000000"
        self.btn_test["command"] = self.test
        self.btn_test.pack(side=tk.TOP, pady=(0, 10))
        

        self.btn_test1 = tk.Button(self)
        self.btn_test1["text"] = "TEST - Fixation"
        self.btn_test1["fg"]   = "white"
        self.btn_test1["bg"]   = "#000000"
        self.btn_test1["command"] = self.test_fixation
        self.btn_test1.pack(side=tk.TOP, pady=(0, 10))
        

        self.btn_test2 = tk.Button(self)
        self.btn_test2["text"] = "TEST - Pursuit (linear)"
        self.btn_test2["fg"]   = "white"
        self.btn_test2["bg"]   = "#000000"
        self.btn_test2["command"] = lambda t="linear": self.test_pursuit(t)
        self.btn_test2.pack(side=tk.TOP, pady=(0, 10))
        

        self.btn_test3 = tk.Button(self)
        self.btn_test3["text"] = "TEST - Pursuit (spiral)"
        self.btn_test3["fg"]   = "white"
        self.btn_test3["bg"]   = "#000000"
        self.btn_test2["command"] = lambda t="spiral": self.test_pursuit(t)
        self.btn_test3.pack(side=tk.TOP, pady=(0, 10))
        
        
        
        
        
        self.btn_default_test = self.make_test_button("Test default", "default")
        self.btn_2p_test = self.make_test_button("Test 2-point", "custom_2p")
        self.btn_5p_test = self.make_test_button("Test 5-point", "custom_5p")
        self.btn_5p_img_test = self.make_test_button("Test 5-point (image)", "custom_5p_img")   
        
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
        self.btn_default_test.pack(side=tk.TOP, pady=(10,10))
        self.btn_2p_test.pack(side=tk.TOP, pady=(0, 10))
        self.btn_5p_test.pack(side=tk.TOP, pady=(0, 10))
        self.btn_5p_img_test.pack(side=tk.TOP, pady=(0, 10))
        self.btn_show_status.pack(side=tk.TOP, pady=(0, 10))
        self.btn_shutdown.pack(side=tk.TOP)
        
        
    def test(self):
        self.controller.make_psycho_window()
        self.custom_calibration(5, "img") 
        self.controller.close_psycho_window()
        
    def test_fixation(self):
        self.controller.make_psycho_window()
#        self.controller.start_fixation_exercise(positions=[(-0.5,-0.5), (0.5,-0.5), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.0)], stimuli_paths=["stimuli/star_yellow.png"])
        self.controller.start_fixation_exercise_animate_transition(positions=[(-0.5,-0.5), (0.5,-0.5), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.0)], stimuli_paths=["stimuli/star_yellow.png"])
        self.controller.close_psycho_window()
        
    def test_pursuit(self, path_type):
        self.controller.make_psycho_window()

        if path_type == "linear":
            self.controller.start_pursuit_exercise(pathing="linear", positions=[(-0.5,-0.5), (0.3, 0.5), (0.5, -0.5), (0.0, 0.0)], stimuli_paths=["stimuli/smiley_yellow.png"])
        elif path_type == "spiral":
            self.controller.start_pursuit_exercise(pathing="spiral", positions=[(-0.7,0.0), (0.0, 0.0)], stimuli_paths=["stimuli/smiley_yellow.png"])
            
        self.controller.close_psycho_window()
        
        
    def test_pursuit_horizontal(self):
        self.controller.make_psycho_window()
        self.controller.start_pursuit_exercise(pathing="linear", positions=[(-0.85,-0.5), (-0.5, 0.7), (0.0, 0.7), (0.1, -0.7), (0.6, -0.7), (0.85, 0.5)], stimuli_paths=["stimuli/smiley_yellow.png"])
        self.controller.close_psycho_window()
        
        
    def make_test_button(self, title, cal_type):
        btn = tk.Button(self)
        btn["text"] = title
        btn["fg"]   = "white"
        btn["bg"]   = "#4CAF50"
        btn["command"] = lambda t=cal_type: self.run_test(t)
        return btn
        
    def run_test(self, cal_type):
        self.controller.make_psycho_window()
        
        self.test_folder = "test_" + cal_type + "/"
         
        if cal_type == "default":
            pass
        elif cal_type == "custom_2p":
            self.custom_calibration(2, "img")
        elif cal_type == "custom_5p":
            self.custom_calibration(5, "img")
        elif cal_type == "custom_5p_img":
            self.custom_calibration(5, "img")
        
        self.controller.flash_screen()
        self.controller.start_fixation_exercise_animate_transition(positions=[(-0.5,-0.5), (0.5,-0.5), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.0)], stimuli_paths=["stimuli/smiley_yellow.png"])
        self.store_data("transformation")
        
        self.controller.flash_screen()
 #        self.controller.start_fixation_exercise(positions=[(-0.5,-0.5), (0.5,-0.5), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.0)], stimuli_paths=["stimuli/star_yellow.png"])
        self.controller.start_fixation_exercise_animate_transition(positions=[(-0.5,-0.5), (0.5,-0.5), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.0)], stimuli_paths=["stimuli/star_yellow.png"])
        self.store_data("training_fixation")
        
        self.controller.flash_screen()
        self.controller.start_pursuit_exercise(pathing="linear", positions=[(-0.5,-0.5), (0.3, 0.5), (0.5, -0.5), (0.0, 0.0)], stimuli_paths=["stimuli/smiley_yellow.png"])
#        self.controller.start_pursuit_exercise(pathing="linear", positions=[(-0.85,-0.5), (-0.5, 0.7), (0.0, 0.7), (0.1, -0.7), (0.6, -0.7), (0.85, 0.5)], stimuli_paths=["stimuli/smiley_yellow.png"])
        self.store_data("training_pursuit_linear")
        
        self.controller.flash_screen()
        self.controller.start_pursuit_exercise(pathing="spiral", positions=[(-0.7,0.0), (0.0, 0.0)], stimuli_paths=["stimuli/smiley_yellow.png"])
        self.store_data("training_pursuit_spiral")
        
        self.controller.close_psycho_window(screen = 1)
   
    def show_status(self, screen=1):
        self.controller.show_status(screen=screen)
   
        
    def custom_calibration(self, num_points, stim_type="default"):
        
        # we can only check if there is an existing eye tracking device.
        # this will still fail whenever the device is there but turned off 
        # TobiiProSDK does not support activity checks this for python it seems..
        self.controller.start_custom_calibration(num_points, stim_type=stim_type)
        
    def store_data(self, testname):
        
         # write data to file
        try: # just in case we run exercise before calibration
            os.makedirs(self.session_path + self.test_folder)
        except Exception:
            # directory already exists
            pass
        
        filename = self.session_path + self.test_folder + testname + ".csv"
        
        # PYTHON 2.x
        with open(filename, mode='wb') as gaze_data_file:
        
            field_names = [data for data in self.controller.gaze_params]
            gaze_data_writer = csv.DictWriter(gaze_data_file, fieldnames=field_names, delimiter=";")
            
            gaze_data_writer.writeheader()
            for gaze_data in self.controller.global_gaze_data:
                gaze_data_writer.writerow(gaze_data)   
                                
    def client_exit(self):
        print("Shutting down")
        self.controller.unsubscribe_dict()
        self.quit()


root = tk.Tk()
#root = tk.Toplevel()

# use the next line if you also want to get rid of the titlebar
#root.attributes("-fullscreen", True)

# set focus
#root.lift()
#root.attributes("-topmost", True)

# Set size of window to screen size
root.geometry("800x600")

app = Application(master=root)
app.mainloop()
root.destroy()
