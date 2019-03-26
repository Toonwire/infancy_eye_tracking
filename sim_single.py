# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda



# Run analyse on
type_of_cal = "custom_5p"


# Session to run
session_folder = "2019-03-26 11.45.07"


# Setting path and files
session_path = "session_data/" + session_folder + "/"
test_folder = session_path + "test_" + type_of_cal + "/"
config_filename = session_path + "config.csv"
cal_filename = test_folder + "transformation.csv"




analyzer = gda.GazeDataAnalyzer()
analyzer.setup(config_filename, cal_filename, "dbscan_fixation")
analyzer.analyze(cal_filename, "dbscan_fixation")

training_filename = test_folder + "training_fixation.csv"
analyzer.analyze(training_filename, "dbscan_fixation")

training_filename = test_folder + "training_pursuit_linear.csv"
analyzer.analyze(training_filename, "dbscan_pursuit")

training_filename = test_folder + "training_pursuit_spiral.csv"
analyzer.analyze(training_filename, "dbscan_pursuit")
