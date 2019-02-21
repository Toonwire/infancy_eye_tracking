# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:05:05 2019

@author: s144451
"""

import glob
import matplotlib.pyplot as plt
import pandas as pd
import math

left_display_gaze_point = None
right_display_gaze_point = None






for csv_file in glob.glob('session_data/config_files/*.csv'):
    
    data_frame = pd.read_csv(csv_file, delimiter=";")
    
    data_frame = pd.read_csv(data_frame['Gaze data filename'][0], delimiter=';')
    
    data_frame = data_frame[(data_frame['left_gaze_point_on_display_area'] != '(nan, nan)')]
    data_frame = data_frame[(data_frame['right_gaze_point_on_display_area'] != '(nan, nan)')]
    
    N = len(data_frame)
    
    left_display_gaze_point = [eval(coord) for coord in data_frame['left_gaze_point_on_display_area']]
    right_display_gaze_point = [eval(coord) for coord in data_frame['right_gaze_point_on_display_area']]
    target_points = [eval(coord) for coord in data_frame['current_target_point_on_display_area']]

    x_left, y_left = map(list, zip(*left_display_gaze_point))
    x_right, y_right = map(list, zip(*right_display_gaze_point))
    x_target, y_target = map(list, zip(*target_points))
    
    plt.scatter(x_left, y_left, marker='x', color='red')
    plt.scatter(x_right, y_right, marker='o', color='green')
    plt.scatter(x_target, y_target, marker='^', color='black')
        
    plt.gca().invert_yaxis()
    plt.gca().xaxis.tick_top()
    plt.show()
    
    left_error = []
    right_error = []
    
    for i in range(0,N):
        target_point = target_points[i]
        
        gaze_point_left = (x_left[i], y_left[i])
        dist_left = math.sqrt(math.pow(gaze_point_left[0] - target_point[0], 2) + math.pow(gaze_point_left[1] - target_point[1], 2))
        left_error.append(dist_left)
        
        gaze_point_right = (x_right[i], y_right[i])
        dist_right = math.sqrt(math.pow(gaze_point_right[0] - target_point[0], 2) + math.pow(gaze_point_right[1] - target_point[1], 2))
        right_error.append(dist_right)
    
    
    plt.plot(range(0,N), left_error, color='red')
    plt.plot(range(0,N), right_error, color='green')
    plt.ylim(0,1)
    plt.show()
   


    