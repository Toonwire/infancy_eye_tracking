# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda



# Run analyse on
type_of_cal = "custom_5p"

# Session to run
#session_folder = "2019-03-29 10.41.18"
session_folder = "infant_noel_5m"


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
#analyzer.analyze(training_filename, filtering_method, "fixation")

training_filename = test_folder + "training_pursuit_linear.csv"
analyzer.analyze(training_filename, "dbscan_pursuit")
##analyzer.analyze(training_filename, filtering_method, "pursuit")

training_filename = test_folder + "training_pursuit_spiral.csv"
analyzer.analyze(training_filename, "dbscan_pursuit")

#import data_correction as cor
#import numpy as np
#targets = np.array([[0.37, 0.31, 0.3, 0.9], [0.91, 0.8, 0.8, 0.2]])
#correction = cor.DataCorrection(targets, 1920, 1080)
#data = np.array([[0.4, 0.35, 0.12, 0.89], [0.98, 0.66, 0.82, 0.13]])
#pixels = correction.norm_to_pixels(data)
#print(pixels)
#print(correction.pixels_to_norm(pixels))