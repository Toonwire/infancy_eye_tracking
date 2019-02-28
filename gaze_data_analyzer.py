# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:05:05 2019

@author: s144451
"""

import matplotlib.pyplot as plt
import pandas as pd
import math
import data_correction as dc
import numpy as np

class GazeDataAnalyzer:

    def read_data(self, filename):
        # read config csv file
        data_frame = pd.read_csv(filename, delimiter=";")
        
        # check for corrupted/missing data in data frames
        data_frame = data_frame[(data_frame['left_gaze_point_on_display_area'] != '(nan, nan)')]
        data_frame = data_frame[(data_frame['right_gaze_point_on_display_area'] != '(nan, nan)')]
        
        # note number of data rows in csv file
        self.N = len(data_frame)
        
        # fetch gaze points from data
        gaze_data_left = np.transpose(np.array([eval(coord) for coord in data_frame['left_gaze_point_on_display_area']]))
        gaze_data_right = np.transpose(np.array([eval(coord) for coord in data_frame['right_gaze_point_on_display_area']]))
        target_points = np.transpose(np.array([eval(coord) for coord in data_frame['current_target_point_on_display_area']]))

        return (gaze_data_left, gaze_data_right, target_points)
    
    
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
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye(gaze_data_left)
        self.data_correction.calibrate_right_eye(gaze_data_right)
        
    
    def analyze(self, training_filename):   
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
        #------------------------------#
        
        ### error analysis - corrected
        self.analyze_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
        
    def analyze_errors(self, gaze_data_left, gaze_data_right, target_points):
        # compute pixel deviations from fixation to target
        pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
        
        # compute euclidean pixel distance from fixation to target
        pixel_dist_err_left = [self.euclid_dist(err[0], err[1]) for err in pixel_err_left]
        pixel_dist_err_right = [self.euclid_dist(err[0], err[1]) for err in pixel_err_right]
        
        # compute how much visual angle error the pixel errors correspond to
        angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
                
        self.plot_scatter(gaze_data_left, gaze_data_right, target_points, title_string="Scatter plot for fixations")
        self.plot_pixel_errors(pixel_dist_err_left, pixel_dist_err_right, title_string="Pixel distance error")
        self.plot_angle_errors(angle_err_left, angle_err_right, title_string="Visual angle error")
        
        
    def euclid_dist(self, a, b):
        return (a**2 + b**2) ** 0.5
    
    
    def compute_pixel_errors(self, gaze_data_left, gaze_data_right, target_points):
        pixel_err_left = [(abs(fix_x-tar_x), abs(fix_y-tar_y)) for fix_x, fix_y, tar_x, tar_y in zip(gaze_data_left[0,:], gaze_data_left[1,:], target_points[0,:], target_points[1,:])]
        pixel_err_right = [(abs(fix_x-tar_x), abs(fix_y-tar_y)) for fix_x, fix_y, tar_x, tar_y in zip(gaze_data_right[0,:], gaze_data_right[1,:], target_points[0,:], target_points[1,:])]
        return (pixel_err_left, pixel_err_right)
    
    def compute_visual_angle_error(self, pixel_err_left_norm, pixel_err_right_norm):
        
        visual_angle_err_left = []
        visual_angle_err_right = []
        
        for err_left_norm, err_right_norm in zip(pixel_err_left_norm, pixel_err_right_norm):
            # convert normalized coordinates to pixel coordinates (as on screen)
            px_err_left_x = err_left_norm[0] * self.screen_width_px
            px_err_left_y = err_left_norm[1] * self.screen_height_px
            px_err_right_x = err_right_norm[0] * self.screen_width_px
            px_err_right_y = err_right_norm[1] * self.screen_height_px            
            
            # calculate distance from target to fixation of left eye in pixels on screen
            px_err_left = self.euclid_dist(px_err_left_x, px_err_left_y)
            px_err_right = self.euclid_dist(px_err_right_x, px_err_right_y)
            
            # calculate visual angle error from pixel error
            visual_angle_err_left_degrees = math.atan(px_err_left/(self.dist_to_screen_cm * self.ppcm)) * 180 / math.pi
            visual_angle_err_right_degrees = math.atan(px_err_right/(self.dist_to_screen_cm * self.ppcm)) * 180 / math.pi
        
            visual_angle_err_left.append(visual_angle_err_left_degrees)
            visual_angle_err_right.append(visual_angle_err_right_degrees)
            
        return (visual_angle_err_left, visual_angle_err_right)
        
    def plot_scatter(self, gaze_data_left, gaze_data_right, targets, title_string=""):
        
        x_left = gaze_data_left[0,:]
        y_left = gaze_data_left[1,:]
        x_right = gaze_data_right[0,:]
        y_right = gaze_data_right[1,:]
        x_targets = targets[0,:]
        y_targets = targets[1,:]
        
        scatter_left = plt.scatter(x_left, y_left, marker='x', color='red')
        scatter_right = plt.scatter(x_right, y_right, marker='o', color='green')
        scatter_target = plt.scatter(x_targets, y_targets, marker='^', color='black')
            
        plt.legend((scatter_left, scatter_right, scatter_target),
                   ("left eye", "right eye", "target points"))
        plt.title(title_string, y=1.08)
        plt.gca().xaxis.tick_top()
        plt.xlim(0,1)
        plt.ylim(1,0)
        plt.show()
        
        
    def plot_angle_errors(self, err_left, err_right, title_string=""):
        self.plot_pixel_errors(err_left, err_right, title_string, y_max=6)
        
    
    def plot_pixel_errors(self, err_left, err_right, title_string="", y_max=1):
        plot_left, = plt.plot(range(0,self.N), err_left, color='red', label="left eye")
        plot_right, = plt.plot(range(0,self.N), err_right, color='green', label="right eye")
        plt.legend(handles=[plot_left, plot_right])
        plt.title(title_string)
        plt.ylim(0,y_max)
        plt.show()






    