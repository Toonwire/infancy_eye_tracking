# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda



# Run analyse on
type_of_cal = "default"

# Filtering data by
filtering_method = "dbscan"
filtering_training_type = "pursuit"

# Session to run
session_folder = "2019-03-19 13.25.37"


# Setting path and files
session_path = "session_data/" + session_folder + "/"
test_folder = session_path + "test_" + type_of_cal + "/"
config_filename = session_path + "config.csv"
cal_filename = test_folder + "transformation.csv"


analyzer = gda.GazeDataAnalyzer()
analyzer.setup(config_filename, cal_filename, "dbscan", "fixation")
analyzer.analyze(cal_filename, "dbscan", "fixation")

training_filename = test_folder + "training_fixation.csv"
analyzer.analyze(training_filename, filtering_method, filtering_training_type)

training_filename = test_folder + "training_pursuit_linear.csv"
analyzer.analyze(training_filename, filtering_method, filtering_training_type)

training_filename = test_folder + "training_pursuit_spiral.csv"
analyzer.analyze(training_filename, filtering_method, filtering_training_type)

