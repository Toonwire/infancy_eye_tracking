# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda



# Run analyse on
type_of_cal = "default"
#type_of_cal = "custom_2p"
#type_of_cal = "custom_5p"

# Session to run
session_folder = "ctrl_group_2_louise"
#session_folder = "infant_d25_gudrun_5m"


# Setting path and files
session_path = "session_data/" + session_folder + "/"
test_folder = session_path + "test_" + type_of_cal + "/"
config_filename = session_path + "config.csv"
cal_filename = test_folder + "training_fixation.csv"



analyzer = gda.GazeDataAnalyzer()

print("\nTEST DATA - FIXATION")
training_filename = test_folder + "training_fixation.csv"
analyzer.pattern_recognition(training_filename, "dbscan_fixation")

print("\nTEST DATA - PURSUIT (CIRCLE)")
training_filename = test_folder + "training_pursuit_circle.csv"
analyzer.pattern_recognition(training_filename, "dbscan_pursuit")

print("\nTEST DATA - PURSUIT (LINEAR)")
training_filename = test_folder + "training_pursuit_linear.csv"
analyzer.pattern_recognition(training_filename, "dbscan_pursuit")

print("\nTEST DATA - PURSUIT (SPIRAL)")
training_filename = test_folder + "training_pursuit_spiral.csv"
analyzer.pattern_recognition(training_filename, "dbscan_pursuit")
