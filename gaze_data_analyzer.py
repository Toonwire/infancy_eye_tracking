# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:05:05 2019

@author: s144451
"""

import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import sys
import data_correction as dc
import dbscan
import numpy as np

class GazeDataAnalyzer:
    
    plt.rcParams.update({'font.size': 12})

    show_graphs_bool = False
    show_rms_pixel_bool = True
    show_rms_degree_bool = True
    
    to_closest_target = False
    
    def read_data(self, filename, filtering_method, remove_nan_values = True):
        # read config csv file
        data_frame = pd.read_csv(filename, delimiter=";")
        
        if remove_nan_values:
            # check for corrupted/missing data in data frames
            data_frame = data_frame[(data_frame['left_gaze_point_on_display_area'] != '(nan, nan)')]
            data_frame = data_frame[(data_frame['right_gaze_point_on_display_area'] != '(nan, nan)')]
            data_frame = data_frame[(data_frame['left_gaze_point_validity'] != 0)]
            data_frame = data_frame[(data_frame['right_gaze_point_validity'] != 0)]
        
        # note number of data rows in csv file
        self.N = len(data_frame)
        
        # fetch gaze points from data
        gaze_data_left_temp = np.transpose(np.array([eval(coord) if coord != "(nan, nan)" else (-1,-1) for coord in data_frame['left_gaze_point_on_display_area']]))
        gaze_data_right_temp = np.transpose(np.array([eval(coord) if coord != "(nan, nan)" else (-1,-1) for coord in data_frame['right_gaze_point_on_display_area']]))
        target_points_temp = np.transpose(np.array([eval(coord) if coord != "(nan, nan)" else (-1,-1) for coord in data_frame['current_target_point_on_display_area']]))

        return self.filtering(filtering_method, gaze_data_left_temp, gaze_data_right_temp, target_points_temp)
    
    def filtering(self, filtering_method, gaze_data_left_temp, gaze_data_right_temp, target_points_temp):
        
        gaze_data_left = []
        gaze_data_right = []
        target_points = []
        
        if filtering_method == "dbscan_fixation" or filtering_method == "dbscan_pursuit":
            
            gaze_data_temp = np.mean(np.array([gaze_data_left_temp, gaze_data_right_temp]), axis=0)
            
            db_scan = dbscan.DBScan()
            clusters = db_scan.run_linear(gaze_data_temp.T, 0.05, 10)
            
            
            if self.show_graphs_bool:
                colours = ['black', 'red', 'blue', 'cyan', 'yellow', 'purple', 'green']
                colors = [colours[int(clusters[key]) % len(colours)] for key in clusters.keys()]
                plt.scatter(*zip(*clusters.keys()),c=colors)
                plt.title("DBScan", y=1.08)
                plt.gca().xaxis.tick_top()
                plt.xlim(0,1)
                plt.ylim(1,0)
                plt.show()
            
            gaze_data_left_x = []
            gaze_data_left_y = []
            gaze_data_right_x = []
            gaze_data_right_y = []
            target_points_x = []
            target_points_y = []
            
            prev_target = np.array([-1.0, -1.0])
            prev_cluster = 0
            
            for i in range(self.N):
                current_target = target_points_temp[:,i]
                p = (gaze_data_temp[0, i], gaze_data_temp[1, i])
                
                if p not in clusters:
                    continue;
                    
#                current_cluster = clusters[p]
                
                # Check if current target is a new target, and if the future target is a new target
#                is_new_target = not np.array_equal(current_target, prev_target)

                dif_new_target = 5
                is_past_new_target = False
                if (i - dif_new_target >= 0):
                    is_past_new_target = not np.array_equal(current_target, target_points_temp[:,i-dif_new_target])
                    
                is_future_new_target = False
                if (i + dif_new_target < self.N):
                    is_future_new_target = not np.array_equal(current_target, target_points_temp[:,i+dif_new_target])
                
                # For fixation filtering
                # A gaze point in a cluster has to be filtered away 
                # If the current target point has changed but the gaze point has not, the gaze point can still be in the old cluster, for the old target point
                # This gaze point should be filtered away, since it has a large visual angle error
                # If the current target is different from the previous target, but the previous target is the same as the target before that, a change has been noted
                if clusters[p] == 0 or (filtering_method == "dbscan_fixation" and (is_past_new_target or is_future_new_target)):
                    del clusters[p]
                else:
                    gaze_data_left_x.append(gaze_data_left_temp[0,i])
                    gaze_data_left_y.append(gaze_data_left_temp[1,i])
                    gaze_data_right_x.append(gaze_data_right_temp[0,i])
                    gaze_data_right_y.append(gaze_data_right_temp[1,i])
                    target_points_x.append(target_points_temp[0,i])
                    target_points_y.append(target_points_temp[1,i])
                        
                prev_target = current_target
#                prev_cluster = current_cluster
            
            if self.show_graphs_bool:
                colours = ['black', 'red', 'blue', 'cyan', 'yellow', 'purple', 'green']
                colors = [colours[int(clusters[key]) % len(colours)] for key in clusters.keys()]
                plt.scatter(*zip(*clusters.keys()),c=colors)
                plt.title("DBScan", y=1.08)
                plt.gca().xaxis.tick_top()
                plt.xlim(0,1)
                plt.ylim(1,0)
                plt.show()
            
            gaze_data_left = np.array([gaze_data_left_x, gaze_data_left_y])
            gaze_data_right = np.array([gaze_data_right_x, gaze_data_right_y])
            target_points = np.array([target_points_x, target_points_y])


        # Remove all points after a shift of target for a half second (45 measures)
        elif filtering_method == "threshold_time_fixation":
            prev_target = np.array([0.0, 0.0])
            wait = 0
            
            gaze_data_left_x = []
            gaze_data_left_y = []
            gaze_data_right_x = []
            gaze_data_right_y = []
            target_points_x = []
            target_points_y = []
            
            for i in range(self.N):
                current_target = target_points_temp[:,i]
                
                if np.array_equal(current_target, prev_target):
                    wait += 1
                    
                    if (wait > 45):
                        gaze_data_left_x.append(gaze_data_left_temp[0,i])
                        gaze_data_left_y.append(gaze_data_left_temp[1,i])
                        gaze_data_right_x.append(gaze_data_right_temp[0,i])
                        gaze_data_right_y.append(gaze_data_right_temp[1,i])
                        target_points_x.append(current_target[0])
                        target_points_y.append(current_target[1])
    
                        
                else:
                    wait = 0
                    
                prev_target = current_target
                
            gaze_data_left = np.array([gaze_data_left_x, gaze_data_left_y])
            gaze_data_right = np.array([gaze_data_right_x, gaze_data_right_y])
            target_points = np.array([target_points_x, target_points_y])

        # Remove all points in the first half second (45 measures)
        elif filtering_method == "threshold_time_pursuit":
            
            wait = 0
            
            gaze_data_left_x = []
            gaze_data_left_y = []
            gaze_data_right_x = []
            gaze_data_right_y = []
            target_points_x = []
            target_points_y = []
            
            for i in range(self.N):
                wait += 1
                
                if (wait > 10):
                    gaze_data_left_x.append(gaze_data_left_temp[0,i])
                    gaze_data_left_y.append(gaze_data_left_temp[1,i])
                    gaze_data_right_x.append(gaze_data_right_temp[0,i])
                    gaze_data_right_y.append(gaze_data_right_temp[1,i])
                    target_points_x.append(target_points_temp[0,i])
                    target_points_y.append(target_points_temp[1,i])

                       
            gaze_data_left = np.array([gaze_data_left_x, gaze_data_left_y])
            gaze_data_right = np.array([gaze_data_right_x, gaze_data_right_y])
            target_points = np.array([target_points_x, target_points_y])
            
        # Do nothing for filter out outliers
        elif filtering_method == None:
            gaze_data_left = gaze_data_left_temp
            gaze_data_right = gaze_data_right_temp
            target_points = target_points_temp
            
        self.N = len(target_points[0,:])
        
        return (gaze_data_left, gaze_data_right, target_points)
    
    def center_by_cluster(self, gaze_data_left, gaze_data_right):
        return (self.data_correction.adjust_by_cluster_center(gaze_data_left), self.data_correction.adjust_by_cluster_center(gaze_data_right))
    
    def animate(self, training_filename, filtering_method = None):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, filtering_method, remove_nan_values = False)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#        gaze_data_left_corrected = gaze_data_left
#        gaze_data_right_corrected = gaze_data_right
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected)
    
    
    # set up the transformation matrices 
    def setup(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename, filtering_method)
        
        # filter gaze points, remove outliers
        # then redefine target as those closest to gaze points
        
        if self.to_closest_target:
            gaze_data_left = self.reject_outliers_gaze_only(gaze_data_left)
            gaze_data_right = self.reject_outliers_gaze_only(gaze_data_right)
            
            target_points = self.find_closest_target(target_points, gaze_data_left, gaze_data_right)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye(gaze_data_left)
        self.data_correction.calibrate_right_eye(gaze_data_right)
        
    
    def analyze(self, training_filename, filtering_method = None, output = "points"):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, filtering_method)
        
        if self.to_closest_target:
            target_points = self.find_closest_target(target_points, gaze_data_left, gaze_data_right)        
        
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#        gaze_data_left_corrected = gaze_data_left
#        gaze_data_right_corrected = gaze_data_right
        
        
#        gaze_data_left_corrected, gaze_data_right_corrected = self.center_by_cluster(gaze_data_left_corrected, gaze_data_right_corrected)
        #------------------------------#
        
        ### error analysis - corrected
        self.analyze_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
        ### error analysis - corrected
#        fixations_filtered_left, filtered_targets = self.reject_outliers(gaze_data_left_corrected, target_points)
#        fixations_filtered_right, filtered_targets = self.reject_outliers(gaze_data_right_corrected, target_points)
#        self.analyze_errors(fixations_filtered_left, fixations_filtered_right, target_points)
        
        
        # RMSE values for raw and corrected data (averaged btween left- and right fixations)
        rmse_raw, rmse_cor, rmse_imp = self.show_rms_pixel(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)                
        
#        pixel_err_left, pixel_err_right = self.compute_pixel_errors_to_closest_target(gaze_data_left, gaze_data_right, target_points)
        pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
        angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
        
        pixel_err_left_corrected, pixel_err_right_corrected = self.compute_pixel_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        angle_err_left_corrected, angle_err_right_corrected = self.compute_visual_angle_error(pixel_err_left_corrected, pixel_err_right_corrected)
        
        rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = self.show_rms_degree(angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
        if output == "values":
            return (rmse_deg_raw, rmse_deg_cor, rmse_deg_imp)
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
    
    def shuffle(self, gaze_data_left, gaze_data_right, target_points):
        
        shuffle_indices = random.sample(range(self.N), self.N)

        shuffled_gaze_data_left = [[],[]]
        shuffled_gaze_data_right = [[],[]]
        shuffled_target_points = [[],[]]

        for i in shuffle_indices:
            
            shuffled_gaze_data_left[0].append(gaze_data_left[0,i])
            shuffled_gaze_data_left[1].append(gaze_data_left[1,i])
            shuffled_gaze_data_right[0].append(gaze_data_right[0,i])
            shuffled_gaze_data_right[1].append(gaze_data_right[1,i])
            shuffled_target_points[0].append(target_points[0,i])
            shuffled_target_points[1].append(target_points[1,i])
        
        return (np.array(shuffled_gaze_data_left), np.array(shuffled_gaze_data_right), np.array(shuffled_target_points))
    
    def cross_validation(self, config_file, training_filename, filtering_method = None, output = "points", k = 4):
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        # Read data
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, filtering_method)
        gaze_data_left, gaze_data_right, target_points = self.shuffle(gaze_data_left, gaze_data_right, target_points)
        
        test_size = self.N / k        
        
        best_k_fold = 0
        best_optimize = 0
        
        for i in range(k):
            
            print("")
            print("")
            print("CROSS VALIDATION: " + str(i + 1) + "/" + str(k))
            print("")
            
            indices = np.array(range(1, self.N))
            training_indices = indices[i*test_size:(i+1)*test_size]
            test_indices = np.array([j for j in indices if j not in training_indices])
            
            training_set_left = gaze_data_left[:,training_indices]
            test_set_left = gaze_data_left[:,test_indices]
            
            training_set_right = gaze_data_right[:,training_indices]
            test_set_right = gaze_data_right[:,test_indices]
            
            training_set_target = target_points[:,training_indices]
            test_set_target = target_points[:,test_indices]

            self.data_correction = dc.DataCorrection(training_set_target, self.screen_width_px, self.screen_height_px)
            self.data_correction.calibrate_left_eye(training_set_left)
            self.data_correction.calibrate_right_eye(training_set_right)


            
            #------ correct raw data ------#
            gaze_data_left_corrected = self.data_correction.adjust_left_eye(test_set_left)
            gaze_data_right_corrected = self.data_correction.adjust_right_eye(test_set_right)
        

        
            # RMSE values for raw and corrected data (averaged btween left- and right fixations)
            rmse_raw, rmse_cor, rmse_imp = self.show_rms_pixel(test_set_left, test_set_right, gaze_data_left_corrected, gaze_data_right_corrected, test_set_target)                
        
            pixel_err_left, pixel_err_right = self.compute_pixel_errors(test_set_left, test_set_right, test_set_target)
            angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
        
            pixel_err_left_corrected, pixel_err_right_corrected = self.compute_pixel_errors(gaze_data_left_corrected, gaze_data_right_corrected, test_set_target)
            angle_err_left_corrected, angle_err_right_corrected = self.compute_visual_angle_error(pixel_err_left_corrected, pixel_err_right_corrected)
            
            rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = self.show_rms_degree(angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
    
            if rmse_deg_imp > best_optimize:
                best_optimize = rmse_deg_imp
                best_k_fold = i
                
        print("")
        print("Best k fold, " + str(best_k_fold + 1))
        print("With a given optimization on " + str(best_optimize))
        print("")
        
        indices = np.array(range(1, self.N))
        training_indices = indices[best_k_fold*test_size:(best_k_fold+1)*test_size]
        test_indices = np.array([j for j in indices if j not in training_indices])
        
        training_set_left = gaze_data_left[:,training_indices]
        test_set_left = gaze_data_left[:,test_indices]
        
        training_set_right = gaze_data_right[:,training_indices]
        test_set_right = gaze_data_right[:,test_indices]
        
        training_set_target = target_points[:,training_indices]
        test_set_target = target_points[:,test_indices]

        self.data_correction = dc.DataCorrection(training_set_target, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye(training_set_left)
        self.data_correction.calibrate_right_eye(training_set_right)


    
    # set up the transformation matrices 
    def setup_poly(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename, filtering_method)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye_poly(gaze_data_left)
        self.data_correction.calibrate_right_eye_poly(gaze_data_right)
        
        
        
    def analyze_poly(self, training_filename, filtering_method = None, output = "points"):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, filtering_method)
        
        ### error analysis - raw
#        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye_poly(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye_poly(gaze_data_right)
        #gaze_data_left_corrected, gaze_data_right_corrected = self.center_by_cluster(gaze_data_left_corrected, gaze_data_right_corrected)
        #------------------------------#
        
        ### error analysis - corrected
#        self.analyze_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
        ### error analysis - corrected
#        fixations_filtered_left, filtered_targets = self.reject_outliers(gaze_data_left_corrected, target_points)
#        fixations_filtered_right, filtered_targets = self.reject_outliers(gaze_data_right_corrected, target_points)
#        self.analyze_errors(fixations_filtered_left, fixations_filtered_right, target_points)
        
        # RMSE values for raw and corrected data (averaged btween left- and right fixations)
        rmse_raw, rmse_cor, rmse_imp = self.show_rms_pixel(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)                
        
#        pixel_err_left, pixel_err_right = self.compute_pixel_errors_to_closest_target(gaze_data_left, gaze_data_right, target_points)
        pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
        angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
        
        pixel_err_left_corrected, pixel_err_right_corrected = self.compute_pixel_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        angle_err_left_corrected, angle_err_right_corrected = self.compute_visual_angle_error(pixel_err_left_corrected, pixel_err_right_corrected)
        
        rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = self.show_rms_degree(angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
        if output == "values":
            return (rmse_deg_raw, rmse_deg_cor, rmse_deg_imp)
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
    
   
        
    
     # set up the transformation matrices 
    def setup_regression(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename, filtering_method)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_eyes_regression(gaze_data_left, gaze_data_right)

        
        
    def analyze_regression(self, training_filename, filtering_method = None, output = "points"):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, filtering_method)
        
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye_regression(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye_regression(gaze_data_right)
        
        ### error analysis - corrected
        self.analyze_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
        ### error analysis - corrected
#        fixations_filtered_left, filtered_targets = self.reject_outliers(gaze_data_left_corrected, target_points)
#        fixations_filtered_right, filtered_targets = self.reject_outliers(gaze_data_right_corrected, target_points)
#        self.analyze_errors(fixations_filtered_left, fixations_filtered_right, target_points)
        
        
        # RMSE values for raw and corrected data (averaged btween left- and right fixations)
        rmse_raw, rmse_cor, rmse_imp = self.show_rms_pixel(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)                
        
#        pixel_err_left, pixel_err_right = self.compute_pixel_errors_to_closest_target(gaze_data_left, gaze_data_right, target_points)
        pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
        angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
        
        pixel_err_left_corrected, pixel_err_right_corrected = self.compute_pixel_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        angle_err_left_corrected, angle_err_right_corrected = self.compute_visual_angle_error(pixel_err_left_corrected, pixel_err_right_corrected)
        
        rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = self.show_rms_degree(angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
        if output == "values":
            return (rmse_deg_raw, rmse_deg_cor, rmse_deg_imp)
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
    
    # set up the transformation matrices 
    def setup_seb(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename, filtering_method)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye_seb(gaze_data_left)
        self.data_correction.calibrate_right_eye_seb(gaze_data_right)

#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#
#        self.data_correction.calibrate_left_eye_seb(gaze_data_left_corrected)
#        self.data_correction.calibrate_right_eye_seb(gaze_data_right_corrected)
            
        
        
    def analyze_seb(self, training_filename, filtering_method = None, output = "points"):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, filtering_method)
        
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye_seb_2(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye_seb_2(gaze_data_right)
        
        #gaze_data_left_corrected_2 = self.data_correction.adjust_left_eye_seb_2(gaze_data_left_corrected)
        #gaze_data_right_corrected_2 = self.data_correction.adjust_right_eye_seb_2(gaze_data_right_corrected)
        #------------------------------#
        
        ### error analysis - corrected
        self.analyze_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
#        self.analyze_errors(gaze_data_left_corrected_2, gaze_data_right_corrected_2, target_points)
        
        ### error analysis - corrected
#        fixations_filtered_left, filtered_targets = self.reject_outliers(gaze_data_left_corrected, target_points)
#        fixations_filtered_right, filtered_targets = self.reject_outliers(gaze_data_right_corrected, target_points)
#        self.analyze_errors(fixations_filtered_left, fixations_filtered_right, target_points)
        
        
        # RMSE values for raw and corrected data (averaged btween left- and right fixations)
        rmse_raw, rmse_cor, rmse_imp = self.show_rms_pixel(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)                
        
#        pixel_err_left, pixel_err_right = self.compute_pixel_errors_to_closest_target(gaze_data_left, gaze_data_right, target_points)
        pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
        angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
        
        pixel_err_left_corrected, pixel_err_right_corrected = self.compute_pixel_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        angle_err_left_corrected, angle_err_right_corrected = self.compute_visual_angle_error(pixel_err_left_corrected, pixel_err_right_corrected)
        
        rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = self.show_rms_degree(angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
        if output == "values":
            return (rmse_deg_raw, rmse_deg_cor, rmse_deg_imp)
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
    # set up the transformation matrices 
    def setup_two_layer(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename, filtering_method)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye(gaze_data_left)
        self.data_correction.calibrate_right_eye(gaze_data_right)

        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)

        self.data_correction.calibrate_left_eye_poly(gaze_data_left_corrected)
        self.data_correction.calibrate_right_eye_poly(gaze_data_right_corrected)
            
        
    def analyze_two_layer(self, training_filename, filtering_method = None):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, filtering_method)
        
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
        
        gaze_data_left_corrected_2 = self.data_correction.adjust_left_eye_poly(gaze_data_left_corrected)
        gaze_data_right_corrected_2 = self.data_correction.adjust_right_eye_poly(gaze_data_right_corrected)
        #------------------------------#
        
        ### error analysis - corrected
        self.analyze_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        self.analyze_errors(gaze_data_left_corrected_2, gaze_data_right_corrected_2, target_points)
        
        ### error analysis - corrected
#        fixations_filtered_left, filtered_targets = self.reject_outliers(gaze_data_left_corrected, target_points)
#        fixations_filtered_right, filtered_targets = self.reject_outliers(gaze_data_right_corrected, target_points)
#        self.analyze_errors(fixations_filtered_left, fixations_filtered_right, target_points)
        
        
        # RMSE values for raw and corrected data (averaged btween left- and right fixations)
        self.show_rms_pixel(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)      
        
        pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
        angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
        
        pixel_err_left_corrected, pixel_err_right_corrected = self.compute_pixel_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        angle_err_left_corrected, angle_err_right_corrected = self.compute_visual_angle_error(pixel_err_left_corrected, pixel_err_right_corrected)
        
        pixel_err_left_corrected_2, pixel_err_right_corrected_2 = self.compute_pixel_errors(gaze_data_left_corrected_2, gaze_data_right_corrected_2, target_points)
        angle_err_left_corrected_2, angle_err_right_corrected_2 = self.compute_visual_angle_error(pixel_err_left_corrected_2, pixel_err_right_corrected_2)

        rmse_deg_raw = (self.rmse_deg(angle_err_left) + self.rmse_deg(angle_err_right)) / 2
        rmse_deg_cor = (self.rmse_deg(angle_err_left_corrected) + self.rmse_deg(angle_err_right_corrected)) / 2
        rmse_deg_cor_2 = (self.rmse_deg(angle_err_left_corrected_2) + self.rmse_deg(angle_err_right_corrected_2)) / 2
        
        print("RMS error raw (deg of visual angle):\t\t" + str(rmse_deg_raw))
        print("RMS error corrected (deg of visual angle):\t" + str(rmse_deg_cor))
        print("RMS error corrected 2 (deg of visual angle):\t" + str(rmse_deg_cor_2))
        print("Change:\t\t\t" + str((rmse_deg_raw - rmse_deg_cor) / max(rmse_deg_raw, rmse_deg_cor) * 100) + " %")
        print("Change 2:\t\t\t" + str((rmse_deg_raw - rmse_deg_cor_2) / max(rmse_deg_raw, rmse_deg_cor_2) * 100) + " %")
#        self.show_rms_degree(angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
    
    def show_rms_pixel(self, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points):
        rmse_raw = (self.rmse(gaze_data_left, target_points) + self.rmse(gaze_data_right, target_points)) / 2
        rmse_cor = (self.rmse(gaze_data_left_corrected, target_points) + self.rmse(gaze_data_right_corrected, target_points)) / 2
        rmse_imp = (rmse_raw - rmse_cor) / max(rmse_raw, rmse_cor) * 100
        
        if self.show_rms_pixel_bool:
            
            print("RMS error raw:\t\t" + str(rmse_raw))
            print("RMS error corrected:\t" + str(rmse_cor))
            print("Change:\t\t\t" + str(rmse_imp) + " %")
    
        return (rmse_raw, rmse_cor, rmse_imp)
    
    
    def show_rms_degree(self, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected):
        rmse_deg_raw = (self.rmse_deg(angle_err_left) + self.rmse_deg(angle_err_right)) / 2
        rmse_deg_cor = (self.rmse_deg(angle_err_left_corrected) + self.rmse_deg(angle_err_right_corrected)) / 2
        rmse_deg_imp = (rmse_deg_raw - rmse_deg_cor) / max(rmse_deg_raw, rmse_deg_cor) * 100
        if self.show_rms_degree_bool:
            
            print("RMS error raw (deg of visual angle):\t\t" + str(rmse_deg_raw))
            print("RMS error corrected (deg of visual angle):\t" + str(rmse_deg_cor))
            print("Change:\t\t\t" + str(rmse_deg_imp) + " %")
            
        return (rmse_deg_raw, rmse_deg_cor, rmse_deg_imp)
    
    def rmse(self, fixations, targets):
        fixations_filtered, filtered_targets = self.reject_outliers(fixations, targets)
        return np.sqrt(((fixations_filtered - filtered_targets) ** 2).mean())

    def rmse_deg(self, degrees):
        degrees_filtered = degrees[abs(degrees - np.mean(degrees)) < 2 * np.std(degrees)]
        return np.sqrt((degrees_filtered ** 2).mean())
    
    def reject_outliers(self, data, targets, m=1.5):
        filtered_x = [x if abs(x - np.mean(data[0,:])) < m * np.std(data[0,:]) else sys.maxint for x in data[0,:]]
        filtered_y = [y if abs(y - np.mean(data[1,:])) < m * np.std(data[1,:]) else sys.maxint for y in data[1,:]]
        
        filtered_data = [[],[]]
        filtered_targets = [[],[]]
        
        for x,y,tx,ty in zip(filtered_x, filtered_y, targets[0,:], targets[1,:]):
            if x != sys.maxint and y != sys.maxint:
                filtered_data[0].append(x)
                filtered_data[1].append(y)
                
                filtered_targets[0].append(tx)
                filtered_targets[1].append(ty)
                
        return (np.array(filtered_data), np.array(filtered_targets))
    
    
    def reject_outliers_no_targets(self, data, m=1.5):
        return [x if abs(x - np.mean(data)) < m * np.std(data) else -1 for x in data]
    
    def reject_outliers_gaze_only(self, data, m=1.5):
        filtered_x = [x if abs(x - np.mean(data[0,:])) < m * np.std(data[0,:]) else sys.maxint for x in data[0,:]]
        filtered_y = [y if abs(y - np.mean(data[1,:])) < m * np.std(data[1,:]) else sys.maxint for y in data[1,:]]
        
        filtered_data = [[],[]]
        
        for x,y in zip(filtered_x, filtered_y):
            if x != sys.maxint and y != sys.maxint:
                filtered_data[0].append(x)
                filtered_data[1].append(y)
            
        return np.array(filtered_data)
    
    def find_closest_target(self, target_points, gaze_data_left, gaze_data_right):
        closest_target_points_x = []
        closest_target_points_y = []        
        
        for i, (left_x, left_y, right_x, right_y) in enumerate(zip(gaze_data_left[0,:], gaze_data_left[1,:], gaze_data_right[0,:], gaze_data_right[1,:])):
            
            min_euclid = 2
            closest_target_x = 1
            closest_target_y = 1
            
            
            for j, (tar_x, tar_y) in enumerate(zip(target_points[0,:], target_points[1,:])):
                
                dist_left_x = abs(left_x - tar_x)
                dist_left_y = abs(left_y - tar_y)
                dist_right_x = abs(right_x - tar_x)
                dist_right_y = abs(right_y - tar_y)
                
                euclid_left = self.euclid_dist(dist_left_x, dist_left_y)
                euclid_right = self.euclid_dist(dist_right_x, dist_right_y)
                
                if euclid_left + euclid_right < min_euclid:
                    min_euclid = euclid_left + euclid_right
                    closest_target_x = tar_x
                    closest_target_y = tar_y
                    
            closest_target_points_x.append(closest_target_x)
            closest_target_points_y.append(closest_target_y)
                    
        return np.array([closest_target_points_x, closest_target_points_y])
    
    def analyze_errors(self, gaze_data_left, gaze_data_right, target_points):
        if self.show_graphs_bool:
    
            # compute pixel deviations from fixation to target
            pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
            
            # compute euclidean pixel distance from fixation to target (NORMALIZED)
            pixel_dist_err_left = [self.euclid_dist(err[0], err[1]) for err in pixel_err_left]
            pixel_dist_err_right = [self.euclid_dist(err[0], err[1]) for err in pixel_err_right]
           
    
            # compute how much visual angle error the pixel errors correspond to
            angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
            
            
                    
            self.plot_scatter(gaze_data_left, gaze_data_right, target_points, title_string="Scatter plot for fixations")
            self.plot_pixel_errors(pixel_dist_err_left, pixel_dist_err_right, title_string="Pixel distance error")
            self.plot_angle_errors(angle_err_left, angle_err_right, title_string="Visual angle error")
            # self.plot_gaze_points_in_pixels(gaze_data_left, gaze_data_right, target_points, title_string="Gaze data on screen")
         
        
        
            
        
    def euclid_dist(self, a, b):
        return (a**2 + b**2) ** 0.5
    
    def compute_pixel_errors_to_closest_target(self, gaze_data_left, gaze_data_right, target_points):
        pixel_err_left = []
        pixel_err_right = []
        
        diff = []
        
        for i, (left_x, left_y, right_x, right_y) in enumerate(zip(gaze_data_left[0,:], gaze_data_left[1,:], gaze_data_right[0,:], gaze_data_right[1,:])):
            
            min_euclid = 2
            
            min_left_x = 1
            min_left_y = 1
            min_right_x = 1
            min_right_y = 1
            
            min_j = 0
            
            for j, (tar_x, tar_y) in enumerate(zip(target_points[0,:], target_points[1,:])):
                
                dist_left_x = abs(left_x - tar_x)
                dist_left_y = abs(left_y - tar_y)
                dist_right_x = abs(right_x - tar_x)
                dist_right_y = abs(right_y - tar_y)
                
                euclid_left = self.euclid_dist(dist_left_x, dist_left_y)
                euclid_right = self.euclid_dist(dist_right_x, dist_right_y)
                
                if euclid_left + euclid_right < min_euclid:
                    min_euclid = euclid_left + euclid_right
                    min_left_x = dist_left_x
                    min_left_y = dist_left_y
                    
                    min_right_x = dist_right_x
                    min_right_y = dist_right_y
                    
                    min_j = j
                    
            pixel_err_left.append((min_left_x, min_left_y))
            pixel_err_right.append((min_right_x, min_right_y))
            
            diff.append(i - min_j)
            
        print("")
        print("Latency for eye to target: " + str(np.mean(diff)))
        print("")
        
        return (pixel_err_left, pixel_err_right)
    
    
    def compute_pixel_errors(self, gaze_data_left, gaze_data_right, target_points):
        pixel_err_left = [(abs(fix_x-tar_x), abs(fix_y-tar_y)) for fix_x, fix_y, tar_x, tar_y in zip(gaze_data_left[0,:], gaze_data_left[1,:], target_points[0,:], target_points[1,:])]
        pixel_err_right = [(abs(fix_x-tar_x), abs(fix_y-tar_y)) for fix_x, fix_y, tar_x, tar_y in zip(gaze_data_right[0,:], gaze_data_right[1,:], target_points[0,:], target_points[1,:])]
        return (pixel_err_left, pixel_err_right)
    
    
    def compute_pixel_errors_as_on_screen(self, gaze_data_left, gaze_data_right, target_points):
        pixel_err_left = [(fix_x-tar_x, fix_y-tar_y) for fix_x, fix_y, tar_x, tar_y in zip(gaze_data_left[0,:], gaze_data_left[1,:], target_points[0,:], target_points[1,:])]
        pixel_err_right = [(fix_x-tar_x, fix_y-tar_y) for fix_x, fix_y, tar_x, tar_y in zip(gaze_data_right[0,:], gaze_data_right[1,:], target_points[0,:], target_points[1,:])]
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
            
        return (np.array(visual_angle_err_left), np.array(visual_angle_err_right))
    
          
  
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
        
    def plot_gaze_points_in_pixels(self, gaze_data_left, gaze_data_right, target_points, title_string=""):
        # the gaze data recorded is normalized
        # flip y-coordinates to turn recording coordinate system (origo in top-left) into screen coordinate system (origo in bottom-left)
        px_left_x = gaze_data_left[0,:] * self.screen_width_px
        px_left_y = self.screen_height_px - gaze_data_left[1,:] * self.screen_height_px
        px_right_x = gaze_data_right[0,:] * self.screen_width_px
        px_right_y = self.screen_height_px - gaze_data_right[1,:] * self.screen_height_px
        
        px_target_x = target_points[0,:] * self.screen_width_px
        px_target_y = self.screen_height_px - target_points[1,:] * self.screen_height_px
                
        
#        px_left = np.array([px_left_x, px_left_y])
#        px_right = np.array([px_right_x, px_right_y])        
#        px_avg = (px_left + px_right)/2
        
        
        ## PLOT LEFT EYE AND TARGETS
        scatter_left = plt.scatter(px_left_x, px_left_y, marker='x', color="red")
        scatter_targets = plt.scatter(px_target_x, px_target_y, marker='o', color="black")
        # plot lines from targets to gaze points
#        for i in range(len(px_target_x)):
#            plt.plot([px_target_x[i], px_left_x[i]],[px_target_y[i], px_left_y[i]], 'k-')
        plt.xlim(0,self.screen_width_px)
        plt.ylim(0,self.screen_height_px)
        plt.title("Left eye gaze data as on screen")
        plt.xlabel("Screen width (pixels)")
        plt.ylabel("Screen height (pixels)")
        plt.legend((scatter_left, scatter_targets),
                   ("left eye", "target points"))
        plt.show()
        
        
        fig = plt.figure(1, figsize=(18,12))
        subplot_gaze_pixels = fig.add_subplot(2,2,2)
        subplot_vertical_err = fig.add_subplot(2,2,4)
        subplot_horizontal_err = fig.add_subplot(2,2,1)
        
#        plt.rcParams.update({'font.size': 12})
        ## PLOT RIGHT EYE AND TARGETS
        scatter_right = subplot_gaze_pixels.scatter(px_right_x, px_right_y, marker='x', color="green")
        scatter_targets = subplot_gaze_pixels.scatter(px_target_x, px_target_y, marker='o', color="black")
        # plot lines from targets to gaze points
#        for i in range(len(px_target_x)):
#            plt.plot([px_target_x[i], px_left_x[i]],[px_target_y[i], px_left_y[i]], 'k-')
        subplot_gaze_pixels.set_xlim(0,self.screen_width_px)
        subplot_gaze_pixels.set_ylim(0,self.screen_height_px)
#        subplot_gaze_pixels.set_title("Right eye gaze data as on screen")
        subplot_gaze_pixels.set_xlabel("Screen width (pixels)")
        subplot_gaze_pixels.set_ylabel("Screen height (pixels)")
        subplot_gaze_pixels.legend((scatter_right, scatter_targets),
                   ("right eye", "target points"))
#        plt.show()
        
        
        ## CAlCULATE VERTICAL ERRORS AS GAZE VARIES HORIZONTALLY
        # convert normalized coordinates to pixel coordinates (as on screen)
        pixel_err_left, pixel_err_right = self.compute_pixel_errors_as_on_screen(gaze_data_left, gaze_data_right, target_points)
        px_err_left_x = []
        px_err_left_y = []
        px_err_right_x = []
        px_err_right_y = []
        for err_left_norm, err_right_norm in zip(pixel_err_left, pixel_err_right):
            px_err_left_x.append(err_left_norm[0] * self.screen_width_px)
            px_err_left_y.append(err_left_norm[1] * self.screen_height_px)
            px_err_right_x.append(err_right_norm[0] * self.screen_width_px)
            px_err_right_y.append(err_right_norm[1] * self.screen_height_px) 
            
#        px_err_left_x, px_left_y = self.data_correction.reject_outliers(px_err_left_x, px_left_y)
#        px_err_left_y, px_left_x = self.data_correction.reject_outliers(px_err_left_y, px_left_x)
#        px_err_right_x, px_right_y = self.data_correction.reject_outliers(px_err_right_x, px_right_y)
#        px_err_right_y, px_right_x = self.data_correction.reject_outliers(px_err_right_y, px_right_x)
        
        # fit a qudratic line for the vertical errors
#        poly_left_x = np.poly1d(np.polyfit(px_left_x, px_err_left_y, 2))
#        poly_left_y = np.poly1d(np.polyfit(px_left_y, px_err_left_x, 2))
#        poly_right_x = np.poly1d(np.polyfit(px_right_x, px_err_right_y, 2))
#        poly_right_y = np.poly1d(np.polyfit(px_right_y, px_err_right_x, 2))
        
        
        poly_left_x = np.poly1d(np.polyfit(px_left_x, px_err_left_y, 2))
        poly_left_y = np.poly1d(np.polyfit(px_left_y, px_err_left_x, 2))
        poly_right_x, det_right_x = self.polyfit(px_right_x, px_err_right_y, 2)
        poly_right_y, det_right_y = self.polyfit(px_right_y, px_err_right_x, 2)
        
        
        # calculate new x's and y's
        line_y_left_x = poly_left_x(px_left_x)
        line_y_left_y = poly_left_y(px_left_y)
        line_y_right_x = poly_right_x(px_right_x)
        line_y_right_y = poly_right_y(px_right_y)
        
        
        subplot_vertical_err.plot(px_right_x, [-a for a in px_err_right_y], 'o', px_right_x, [-a for a in line_y_right_x]) 
#        subplot_vertical_err.set_title("Right eye vertical error as gaze varies horizontally")
        subplot_vertical_err.set_xlabel("X Coordinate (pixels)")
        subplot_vertical_err.set_ylabel("Vertical error (pixels)")
        subplot_vertical_err.set_xlim(0, self.screen_width_px)
        
        
         
        subplot_horizontal_err.plot(px_err_right_x, px_right_y, 'o', line_y_right_y, px_right_y)  
#        subplot_horizontal_err.set_title("Right eye horizontal error as gaze varies vertically")
        subplot_horizontal_err.set_xlabel("Horizontal error (pixels)")
        subplot_horizontal_err.set_ylabel("Y Coordinate (pixels)")
        subplot_horizontal_err.set_ylim(0,self.screen_height_px)
#        
        plt.show()
        
    
    # Polynomial Regression
    def polyfit(self, x, y, degree):
        coeffs = np.polyfit(x, y, degree)
    
    
        # r-squared
        p = np.poly1d(coeffs)
        # fit values, and mean
        yhat = p(x)                         # or [p(z) for z in x]
        ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
        det = ssreg / sstot
    
        return (p, det)
    
    def pattern_recognition(self, training_filename, filtering_method = None, output = "points"):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, filtering_method)
        
        n = len(target_points[0,:])
        
        target_degrees = []
        gaze_left_degrees = []
        gaze_right_degrees = []
        look_up = 5
        for i in range(n):
            
            start_index = i - look_up
            end_index = i + look_up
            if i < look_up:
                start_index = 0
            if i + 5 > n - 1:
                end_index = n - 1
                
            target_degree = []
            gaze_left_degree = []
            gaze_right_degree = []
            while start_index < end_index:
                
                target_degree.append(self.find_degree(target_points, start_index))
                gaze_left_degree.append(self.find_degree(gaze_data_left, start_index))
                gaze_right_degree.append(self.find_degree(gaze_data_right, start_index))
                start_index += 1
            
            target_degrees.append(np.average(target_degree))
            gaze_left_degrees.append(np.average(gaze_left_degree))
            gaze_right_degrees.append(np.average(gaze_right_degree))
        
        correct_left = 0
        correct_right = 0
        error_left = 0
        error_right = 0
        
        for i in range(n):
            if (target_degrees[i] > 0 and gaze_left_degrees[i] > 0) or (target_degrees[i] < 0 and gaze_left_degrees[i] < 0):
                correct_left += 1
            else:
                error_left += 1
                
            if (target_degrees[i] > 0 and gaze_right_degrees[i] > 0) or (target_degrees[i] < 0 and gaze_right_degrees[i] < 0):
                correct_right += 1
            else:
                error_right += 1
        
        print("Correct left: " + str(correct_left))
        print("Correct right: " + str(correct_right))
        print("Error left: " + str(error_left))
        print("Error right: " + str(error_right))
        
    def find_degree(self, points, index):
        s = points[:,index]
        e = points[:,index + 1]
        
        radians = math.atan2(e[0] - s[0], e[1] - s[1])
        degree = math.degrees(radians)

        return degree
        
            
            
        