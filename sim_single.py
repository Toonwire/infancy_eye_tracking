# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda



# Run analyse on
type_of_cal = "custom_2p"

# Session to run
session_folder = "2019-03-29 10.41.18"


# Setting path and files
session_path = "session_data/" + session_folder + "/"
test_folder = session_path + "test_" + type_of_cal + "/"
config_filename = session_path + "config.csv"
cal_filename = test_folder + "transformation.csv"



analyzer = gda.GazeDataAnalyzer()
analyzer.setup_regression(config_filename, cal_filename)
analyzer.analyze_regression(cal_filename)

training_filename = test_folder + "training_fixation.csv"
analyzer.analyze_regression(training_filename)
#analyzer.analyze(training_filename, filtering_method, "fixation")

training_filename = test_folder + "training_pursuit_linear.csv"
analyzer.analyze_regression(training_filename)
##analyzer.analyze(training_filename, filtering_method, "pursuit")

#training_filename = test_folder + "training_pursuit_spiral.csv"
#analyzer.analyze_regression(training_filename, "threshold_time_pursuit")
#analyzer.analyze(training_filename, filtering_method, "pursuit")
