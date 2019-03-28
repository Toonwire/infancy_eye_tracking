# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda



# Run analyse on
type_of_cal = "default"

# Session to run
session_folder = "ctrl_group_louise-kopi"


# Setting path and files
session_path = "session_data/" + session_folder + "/"
test_folder = session_path + "test_" + type_of_cal + "/"
config_filename = session_path + "config.csv"
cal_filename = test_folder + "transformation.csv"



analyzer = gda.GazeDataAnalyzer()
analyzer.setup_poly(config_filename, cal_filename, "threshold_time_fixation")
analyzer.analyze_poly(cal_filename, "threshold_time_fixation")

training_filename = test_folder + "training_fixation.csv"
analyzer.analyze_poly(training_filename, "threshold_time_fixation")

training_filename = test_folder + "training_pursuit_linear.csv"
analyzer.analyze_poly(training_filename, "threshold_time_pursuit")

#training_filename = test_folder + "training_pursuit_spiral.csv"
#analyzer.analyze_poly(training_filename, "threshold_time_pursuit")
