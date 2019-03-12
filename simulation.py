# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda

analyzer = gda.GazeDataAnalyzer()

session_path = "2019-03-12 13.11.04"
session_path = "session_data/" + session_path + "/"


config_filename = session_path + "config.csv"

cal_file_index = 1
training_file_index = 3


cal_filename = session_path + "calibrations/cal_" + str(cal_file_index) + ".csv"

analyzer.setup(config_filename, cal_filename)
analyzer.analyze(cal_filename)


training_filename = ""
try:
    training_filename = session_path + "training_with_cal_" + str(cal_file_index) + "/fixation_training_" + str(training_file_index) + ".csv"
    analyzer.analyze(training_filename)
except: 
    training_filename = session_path + "training_with_cal_" + str(cal_file_index) + "/pursuit_training_" + str(training_file_index) + ".csv"
    analyzer.analyze(training_filename)



