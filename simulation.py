# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda

analyzer = gda.GazeDataAnalyzer()

session_path = "2019-03-14 10.29.21"
test_type = "5p_img"

session_path = "session_data/" + session_path + "/"
test_folder = session_path + "test_" + test_type + "/"

config_filename = session_path + "config.csv"


cal_filename = test_folder + "transformation.csv"

analyzer.setup(config_filename, cal_filename)
analyzer.analyze(cal_filename)

training_filename = test_folder + "training_fixation.csv"
analyzer.analyze(training_filename)

training_filename = test_folder + "training_pursuit_linear.csv"
analyzer.analyze(training_filename)

training_filename = test_folder + "training_pursuit_spiral.csv"
analyzer.analyze(training_filename)
