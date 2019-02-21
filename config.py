# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 09:25:52 2019

@author: s144451
"""


try:
    # for Python2
    import Tkinter as tk
except ImportError:
    # for Python3
    import tkinter as tk
    
import datetime

licence_file_path = "licenses/license_key_00395217_-_DTU_Compute_IS404-100106342114"
gazedata_file_path = "gaze_data/"+datetime.datetime.now().strftime("%A, %d. %B %Y %I.%M.%S %p")+".csv"

root = tk.Tk()
min_x = 0.0
max_x = float(root.winfo_screenwidth())

min_y = 0.0
max_y = float(root.winfo_screenheight())

target_centers = [(60.0,60.0), (max_x - 60.0, max_y - 60.0)]

