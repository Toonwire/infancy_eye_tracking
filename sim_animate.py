# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda
from psychopy_tobii_controller.tobii_wrapper import tobii_controller

# Run analyse on
type_of_cal = "default"

# Session to run
session_folder = "infant_noel_5m"
session_folder = "2019-03-19 13.17.10"


# Setting path and files
session_path = "session_data/" + session_folder + "/"
test_folder = session_path + "test_" + type_of_cal + "/"
config_filename = session_path + "config.csv"
cal_filename = test_folder + "transformation.csv"



analyzer = gda.GazeDataAnalyzer()

print("\nSETUP TRANSFORMATION")
analyzer.setup(config_filename, cal_filename, "dbscan_fixation")

print("\nRUN ANIMATION")
#training_filename = test_folder + "transformation.csv"  # use "dbscan_fixation"
#training_filename = test_folder + "training_fixation.csv"  # use "dbscan_fixation"
training_filename = test_folder + "training_pursuit_linear.csv"  # use "dbscan_pursuit"
#training_filename = test_folder + "training_pursuit_spiral.csv"  # use "dbscan_pursuit"
target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected = analyzer.animate(training_filename)

controller = tobii_controller(analyzer.screen_width_px, analyzer.screen_height_px)
controller.set_dist_to_screen(analyzer.dist_to_screen_cm)   
controller.animate_test(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)