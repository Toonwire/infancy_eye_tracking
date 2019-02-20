# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:05:05 2019

@author: s144451
"""

import csv
import glob
import matplotlib.pyplot as plt
import pandas as pd
import config
import math

left_display_gaze_point = None
right_display_gaze_point = None




for csv_file in glob.glob('gaze_data/*.csv'):
    data_frame = pd.read_csv(csv_file, delimiter=';')
    
    print(len(data_frame))
    
    data_frame = data_frame[(data_frame['left_gaze_point_on_display_area'] != '(nan, nan)')]
    data_frame = data_frame[(data_frame['right_gaze_point_on_display_area'] != '(nan, nan)')]
    
    print(len(data_frame))
    
    left_display_gaze_point = [eval(coord) for coord in data_frame['left_gaze_point_on_display_area']]
    right_display_gaze_point = [eval(coord) for coord in data_frame['right_gaze_point_on_display_area']]

    x_left, y_left = map(list, zip(*left_display_gaze_point))
    x_right, y_right = map(list, zip(*right_display_gaze_point))
    plt.scatter(x_left, y_left, marker='x', color='red')
    plt.scatter(x_right, y_right, marker='o', color='green')
    
    x_targets, y_targets = map(list, zip(*config.target_centers))
    x_targets_norm = [(x - config.min_x) / (config.max_x - config.min_x) for x in map(float, x_targets)]
    y_targets_norm = [(y - config.min_y) / (config.max_y - config.min_y) for y in map(float, y_targets)]
    plt.scatter(x_targets_norm, y_targets_norm, marker='^', color='black')
    
    plt.gca().invert_yaxis()
    plt.gca().xaxis.tick_top()
    plt.show()
    
    
    left_error = []
    right_error = []
    
    for i in range(500,895):
        target_point = (x_targets_norm[0], y_targets_norm[0])
        
        gaze_point_left = (x_left[i], y_left[i])
        dist_left = math.sqrt(math.pow(gaze_point_left[0] - target_point[0], 2) + math.pow(gaze_point_left[1] - target_point[1], 2))
        left_error.append(dist_left)
        
        gaze_point_right = (x_right[i], y_right[i])
        dist_right = math.sqrt(math.pow(gaze_point_right[0] - target_point[0], 2) + math.pow(gaze_point_right[1] - target_point[1], 2))
        right_error.append(dist_right)
    
    for i in range(895,1450):
        target_point = (x_targets_norm[1], y_targets_norm[1])
        
        gaze_point_left = (x_left[i], y_left[i])
        dist_left = math.sqrt(math.pow(gaze_point_left[0] - target_point[0], 2) + math.pow(gaze_point_left[1] - target_point[1], 2))
        left_error.append(dist_left)
        
        gaze_point_right = (x_right[i], y_right[i])
        dist_right = math.sqrt(math.pow(gaze_point_right[0] - target_point[0], 2) + math.pow(gaze_point_right[1] - target_point[1], 2))
        right_error.append(dist_right)
    
    plt.plot(range(500,1450), left_error, color='red')
    plt.plot(range(500,1450), right_error, color='green')
    plt.ylim(0,1)
    plt.show()
   


    