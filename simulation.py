# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda


analyzer = gda.GazeDataAnalyzer()

session_path = "2019-02-25 17.15.05"
session_path = "session_data/" + session_path + "/"

config_filename = session_path + "config.csv"

cal_file_index = 1
training_file_index = 2


cal_filename = session_path + "calibrations/cal_" + str(cal_file_index) + ".csv"
training_filename = session_path + "training_with_cal_" + str(cal_file_index) + "/training_" + str(training_file_index) + ".csv"


analyzer.setup(config_filename, cal_filename)
analyzer.analyze(cal_filename)
#analyzer.analyze(training_filename)