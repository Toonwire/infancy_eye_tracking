# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:05:05 2019

@author: s144451
"""

import glob
import matplotlib.pyplot as plt
import pandas as pd
import math
import data_correction as dc
import numpy as np

left_display_gaze_point = None
right_display_gaze_point = None

class GazeDataAnalyzer:
    
    def compute_visual_angle_error(self, left_errors_norm, right_errors_norm):
        
        #px_center_x = config.PIXEL_WIDTH / 2
        #px_center_y = config.PIXEL_HEIGHT / 2
        
        # calculate PPCM (pixels per cenitmeter) 
        
        # calculate how much 1 degree visual angle correspond to in pixels
        #px_err = math.tan(math.radians(1)) * config.DIST_TO_SCREEN_CM * ppcm
        
        visual_angle_err_left = []
        visual_angle_err_right = []
        
        for left_error_norm, right_error_norm in zip(left_errors_norm, right_errors_norm):
            # convert normalized coordinates to pixel coordinates
            px_err_left_x = left_error_norm[0] * self.screen_width_px
            px_err_left_y = left_error_norm[1] * self.screen_height_px
            
            px_err_right_x = right_error_norm[0] * self.screen_width_px
            px_err_right_y = right_error_norm[1] * self.screen_height_px
            
            
            
            # calculate distance from target to fixation of left eye in pixels on screen
            px_err_left = math.sqrt(math.pow(px_err_left_x, 2) + math.pow(px_err_left_y, 2))
            
            # calculate distance from target to fixation of right eye in pixels on screen
            px_err_right = math.sqrt(math.pow(px_err_right_x, 2) + math.pow(px_err_right_y, 2))
            
            # calculate visual angle error from pixel error
            visual_angle_err_left_radians = math.atan(px_err_left/(self.dist_to_screen_cm * self.ppcm))
            visual_angle_err_left_degrees = visual_angle_err_left_radians * 180 / math.pi
            
            visual_angle_err_right_radians = math.atan(px_err_right/(self.dist_to_screen_cm * self.ppcm))
            visual_angle_err_right_degrees = visual_angle_err_right_radians * 180 / math.pi
        
            visual_angle_err_left.append(visual_angle_err_left_degrees)
            visual_angle_err_right.append(visual_angle_err_right_degrees)
            
            
        return (visual_angle_err_left, visual_angle_err_right)


    # set up the transformation matrices 
    def setup(self, config_file, cal_filename):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        # override current config data frame and replace with gaze data data frame
        data_frame = pd.read_csv(cal_filename, delimiter=';')
        
        data_frame = data_frame[(data_frame['left_gaze_point_on_display_area'] != '(nan, nan)')]
        data_frame = data_frame[(data_frame['right_gaze_point_on_display_area'] != '(nan, nan)')]
        
        N = len(data_frame)
        
        left_display_gaze_point = [eval(coord) for coord in data_frame['left_gaze_point_on_display_area']]
        right_display_gaze_point = [eval(coord) for coord in data_frame['right_gaze_point_on_display_area']]
        target_points = [eval(coord) for coord in data_frame['current_target_point_on_display_area']]
    
        x_left, y_left = map(list, zip(*left_display_gaze_point))
        x_right, y_right = map(list, zip(*right_display_gaze_point))
        x_target, y_target = map(list, zip(*target_points))
        
        fixations_left = np.zeros((2,N))
        fixations_right = np.zeros((2,N))
        targets = np.zeros((2,N))
        
        for i in range(0,N):
            fixations_left[0,i] = x_left[i];
            fixations_left[1,i] = y_left[i];
        
            fixations_right[0,i] = x_right[i];
            fixations_right[1,i] = y_right[i];
        
            targets[0,i] = x_target[i];
            targets[1,i] = y_target[i];
        
        self.data_correction = dc.DataCorrection(targets, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye(fixations_left)
        self.data_correction.calibrate_right_eye(fixations_right)
        
    
    def analyze(self, training_filename):      
        
        # override current config data frame and replace with gaze data data frame
        data_frame = pd.read_csv(training_filename, delimiter=';')
        
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
            
        plt.gca().xaxis.tick_top()
        plt.xlim(0,1)
        plt.ylim(1,0)
        plt.show()
        
        left_error_norm = []
        right_error_norm = []
        left_error = []
        right_error = []
        
        for i in range(0,N):
            target_point = target_points[i]
            
            gaze_point_left = (x_left[i], y_left[i])
            gaze_point_right = (x_right[i], y_right[i])
            
            left_error_norm_single = (abs(gaze_point_left[0] - target_point[0]), abs(gaze_point_left[1] - target_point[1]))
            right_error_norm_single = (abs(gaze_point_right[0] - target_point[0]), abs(gaze_point_right[1] - target_point[1]))
            
            dist_left = math.sqrt(math.pow(left_error_norm_single[0], 2) + math.pow(left_error_norm_single[1], 2))
            left_error.append(dist_left)
            left_error_norm.append(left_error_norm_single)
            
            dist_right = math.sqrt(math.pow(right_error_norm_single[0], 2) + math.pow(right_error_norm_single[1], 2))
            right_error.append(dist_right)
            right_error_norm.append(right_error_norm_single)
        
        
        visual_angle_err_left, visual_angle_err_right = self.compute_visual_angle_error(left_error_norm, right_error_norm)
        
        plt.plot(range(0,N), left_error, color='red')
        plt.plot(range(0,N), right_error, color='green')
        plt.ylim(0,1)
        plt.show()
       
        plt.plot(range(0,N), visual_angle_err_left, color='red')
        plt.plot(range(0,N), visual_angle_err_right, color='green')
        plt.ylim(0,5)
        plt.show()
   
        fixations_left = np.zeros((2,N))
        fixations_right = np.zeros((2,N))
        targets = np.zeros((2,N))
        
        for i in range(0,N):
            fixations_left[0,i] = x_left[i];
            fixations_left[1,i] = y_left[i];
        
            fixations_right[0,i] = x_right[i];
            fixations_right[1,i] = y_right[i];
        
            targets[0,i] = x_target[i];
            targets[1,i] = y_target[i];
            
            
        corrected_gaze_data_left = self.data_correction.adjust_left_eye(fixations_left)
        corrected_gaze_data_right = self.data_correction.adjust_right_eye(fixations_right)
        
        
        x_left_corrected = corrected_gaze_data_left[0, :]
        y_left_corrected = corrected_gaze_data_left[1, :]
        
        x_right_corrected = corrected_gaze_data_right[0, :]
        y_right_corrected = corrected_gaze_data_right[1, :]
        
        visual_scatter_err_plot_left_corrected = plt.scatter(x_left_corrected, y_left_corrected, marker='x', color='red')
        visual_scatter_err_plot_right_corrected = plt.scatter(x_right_corrected, y_right_corrected, marker='o', color='green')
        visual_scatter_err_plot_target_corrected = plt.scatter(x_target, y_target, marker='^', color='black')
            
        plt.legend((visual_scatter_err_plot_left_corrected, visual_scatter_err_plot_right_corrected, visual_scatter_err_plot_target_corrected),
                   ("left eye", "right eye", "target points"))
        plt.title("Scatter plot for corrected data")
        plt.gca().xaxis.tick_top()
        plt.xlim(0,1)
        plt.ylim(1,0)
        plt.show()
        
        left_error_norm_corrected = []
        right_error_norm_corrected = []
        left_error_corrected = []
        right_error_corrected = []
        
        for i in range(0,N):
            target_point = target_points[i]
            
            gaze_point_left_corrected = (x_left_corrected[i], y_left_corrected[i])
            gaze_point_right_corrected = (x_right_corrected[i], y_right_corrected[i])
            
            left_error_norm_single_corrected = (abs(gaze_point_left_corrected[0] - target_point[0]), abs(gaze_point_left_corrected[1] - target_point[1]))
            right_error_norm_single_corrected = (abs(gaze_point_right_corrected[0] - target_point[0]), abs(gaze_point_right_corrected[1] - target_point[1]))
            
            dist_left_corrected = math.sqrt(math.pow(left_error_norm_single_corrected[0], 2) + math.pow(left_error_norm_single_corrected[1], 2))
            left_error_corrected.append(dist_left_corrected)
            left_error_norm_corrected.append(left_error_norm_single_corrected)
            
            dist_right_corrected = math.sqrt(math.pow(right_error_norm_single_corrected[0], 2) + math.pow(right_error_norm_single_corrected[1], 2))
            right_error_corrected.append(dist_right_corrected)
            right_error_norm_corrected.append(right_error_norm_single_corrected)
        
        
        
        visual_dist_err_plot_left_corrected, = plt.plot(range(0,N), left_error_corrected, color='red', label="left eye")
        visual_dist_err_plot_right_corrected, = plt.plot(range(0,N), right_error_corrected, color='green', label="right eye")
        plt.legend(handles=[visual_dist_err_plot_left_corrected, visual_dist_err_plot_right_corrected])
        plt.title("Visual distance error (corrected)")
        plt.ylim(0,1)
        plt.show()
        
        
        
        visual_angle_err_left_corrected, visual_angle_err_right_corrected = self.compute_visual_angle_error(left_error_norm_corrected, right_error_norm_corrected)
       
        visual_angle_err_plot_left_corrected, = plt.plot(range(0,N), visual_angle_err_left_corrected, color='red', label="left eye")
        visual_angle_err_plot_right_corrected, = plt.plot(range(0,N), visual_angle_err_right_corrected, color='green', label="right eye")
        plt.legend(handles=[visual_angle_err_plot_left_corrected, visual_angle_err_plot_right_corrected])
        plt.title("Visual angle error (corrected)")
        plt.ylim(0,5)
        plt.show()










    