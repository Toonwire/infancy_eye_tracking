# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

from psychopy_tobii_controller.tobii_wrapper import tobii_controller
import pandas as pd
import numpy as np

# Run analyse on
type_of_cal = "default"

# Session to run
session_folder = "ctrl_group_2_seb"


# Setting path and files
session_path = "session_data/" + session_folder + "/"
test_folder = session_path + "test_" + type_of_cal + "/"
config_filename = session_path + "config.csv"
data_filename = test_folder + "training_pursuit_all.csv"

data_frame = pd.read_csv(config_filename, delimiter=";")

screen_width_px = data_frame['Screen width (px)'][0]
screen_height_px = data_frame['Screen height (px)'][0]
screen_size_diag_inches = data_frame['Screen size (inches)'][0]
dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
ppcm = (screen_width_px**2 + screen_height_px**2)**0.5 / (screen_size_diag_inches*2.54)



data_frame = pd.read_csv(data_filename, delimiter=";")

## check for corrupted/missing data in data frames
data_frame = data_frame[(data_frame['left_gaze_point_on_display_area'] != '(nan, nan)')]
data_frame = data_frame[(data_frame['right_gaze_point_on_display_area'] != '(nan, nan)')]
data_frame = data_frame[(data_frame['left_gaze_point_validity'] != 0)]
data_frame = data_frame[(data_frame['right_gaze_point_validity'] != 0)]

# fetch gaze points from data
gaze_data_left = np.transpose(np.array([eval(coord) if coord != "(nan, nan)" else (-1,-1) for coord in data_frame['left_gaze_point_on_display_area']]))
gaze_data_right = np.transpose(np.array([eval(coord) if coord != "(nan, nan)" else (-1,-1) for coord in data_frame['right_gaze_point_on_display_area']]))
target_points = np.transpose(np.array([eval(coord) if coord != "(nan, nan)" else (-1,-1) for coord in data_frame['current_target_point_on_display_area']]))


controller = tobii_controller(screen_width_px, screen_height_px)
controller.set_dist_to_screen(dist_to_screen_cm)   
controller.animate_test_2(gaze_data_left, gaze_data_right, target_points, frame_delay = 0.000)