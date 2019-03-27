# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:05:05 2019

@author: s144451
"""

import matplotlib.pyplot as plt
import pandas as pd
import math
import data_correction as dc
import dbscan
import numpy as np



class GazeDataAnalyzer:

    def read_data(self, filename, filtering_method):
        # read config csv file
        data_frame = pd.read_csv(filename, delimiter=";")
        
        # check for corrupted/missing data in data frames
        data_frame = data_frame[(data_frame['left_gaze_point_on_display_area'] != '(nan, nan)')]
        data_frame = data_frame[(data_frame['right_gaze_point_on_display_area'] != '(nan, nan)')]
        data_frame = data_frame[(data_frame['left_gaze_point_validity'] != 0)]
        data_frame = data_frame[(data_frame['right_gaze_point_validity'] != 0)]
        
        # note number of data rows in csv file
        self.N = len(data_frame)
        
        # fetch gaze points from data
        gaze_data_left_temp = np.transpose(np.array([eval(coord) for coord in data_frame['left_gaze_point_on_display_area']]))
        gaze_data_right_temp = np.transpose(np.array([eval(coord) for coord in data_frame['right_gaze_point_on_display_area']]))
        target_points_temp = np.transpose(np.array([eval(coord) for coord in data_frame['current_target_point_on_display_area']]))

        return self.filtering(filtering_method, gaze_data_left_temp,gaze_data_right_temp,target_points_temp)
    
    def filtering(self, filtering_method, gaze_data_left_temp,gaze_data_right_temp,target_points_temp):
        gaze_data_temp = np.mean(np.array([gaze_data_left_temp, gaze_data_right_temp]), axis=0)
        
        gaze_data_left = []
        gaze_data_right = []
        target_points = []
        
        if filtering_method == "dbscan_fixation" or filtering_method == "dbscan_pursuit":
            
            db_scan = dbscan.DBScan()
            clusters = db_scan.run(gaze_data_temp.T, 0.05, 10)
            
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
            clusters_used = set()
            clusters_used.add(0)
            
            for i in range(self.N):
                current_target = target_points_temp[:,i]
                p = (gaze_data_temp[0, i], gaze_data_temp[1, i])
                
                current_cluster = clusters[p]
                
                # For fixation filtering
                # A gaze point in a cluster has to be filtered away 
                # If the current target point has changed but the gaze point has not, the gaze point can still be in the old cluster, for the old target point
                # This gaze point should be filtered away, since it has a large visual angle error
                # If the current target is different from the previous target, but the previous target is the same as the target before that, a change has been noted
                if filtering_method == "dbscan_fixation" and not np.array_equal(current_target, prev_target):
                    clusters_used.add(prev_cluster)
                
                if current_cluster not in clusters_used:
                    gaze_data_left_x.append(gaze_data_left_temp[0,i])
                    gaze_data_left_y.append(gaze_data_left_temp[1,i])
                    gaze_data_right_x.append(gaze_data_right_temp[0,i])
                    gaze_data_right_y.append(gaze_data_right_temp[1,i])
                    target_points_x.append(target_points_temp[0,i])
                    target_points_y.append(target_points_temp[1,i])
            
                prev_target = current_target
                prev_cluster = current_cluster
            
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
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye(gaze_data_left)
        self.data_correction.calibrate_right_eye(gaze_data_right)
        
    
    def analyze(self, training_filename, filtering_method = None):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, filtering_method)
        
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
        #------------------------------#
        
        ### error analysis - corrected
        self.analyze_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
        ### error analysis - corrected
#        fixations_filtered_left, filtered_targets = self.reject_outliers(gaze_data_left_corrected, target_points)
#        fixations_filtered_right, filtered_targets = self.reject_outliers(gaze_data_right_corrected, target_points)
#        self.analyze_errors(fixations_filtered_left, fixations_filtered_right, target_points)
        
        
        # RMSE values for raw and corrected data (averaged btween left- and right fixations)
        rmse_raw = (self.rmse(gaze_data_left, target_points) + self.rmse(gaze_data_right, target_points)) / 2
        rmse_cor = (self.rmse(gaze_data_left_corrected, target_points) + self.rmse(gaze_data_right_corrected, target_points)) / 2
        
        print("RMS error raw:\t\t" + str(rmse_raw))
        print("RMS error corrected:\t" + str(rmse_cor))
        print("Change:\t\t\t" + str((rmse_raw - rmse_cor) / max(rmse_raw, rmse_cor) * 100) + " %")
        
        
        
        pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
        angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
        
        pixel_err_left_corrected, pixel_err_right_corrected = self.compute_pixel_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        angle_err_left_corrected, angle_err_right_corrected = self.compute_visual_angle_error(pixel_err_left_corrected, pixel_err_right_corrected)
        
        rmse_deg_raw = (self.rmse_deg(angle_err_left) + self.rmse_deg(angle_err_right)) / 2
        rmse_deg_cor = (self.rmse_deg(angle_err_left_corrected) + self.rmse_deg(angle_err_right_corrected)) / 2
        
        print("RMS error raw (deg of visual angle):\t\t" + str(rmse_deg_raw))
        print("RMS error corrected (deg of visual angle):\t" + str(rmse_deg_cor))
        print("Change:\t\t\t" + str((rmse_deg_raw - rmse_deg_cor) / max(rmse_deg_raw, rmse_deg_cor) * 100) + " %")
        
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
#        self.data_correction.calibrate_left_eye(gaze_data_left)
#        self.data_correction.calibrate_right_eye(gaze_data_right)

        #gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        #gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)

        #self.fine_data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        #self.fine_data_correction.calibrate_left_eye(gaze_data_left_corrected)
        #self.fine_data_correction.calibrate_right_eye(gaze_data_right_corrected)


        self.data_correction.calibrate_left_eye_seb(gaze_data_left)
        self.data_correction.calibrate_right_eye_seb(gaze_data_right)
        
        
    def analyze_seb(self, training_filename, filtering_method = None):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, filtering_method)
        
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)

        gaze_data_left_corrected = self.data_correction.adjust_left_eye_seb(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye_seb(gaze_data_right)
        #------------------------------#
        
        ### error analysis - corrected
        self.analyze_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
        ### error analysis - corrected
#        fixations_filtered_left, filtered_targets = self.reject_outliers(gaze_data_left_corrected, target_points)
#        fixations_filtered_right, filtered_targets = self.reject_outliers(gaze_data_right_corrected, target_points)
#        self.analyze_errors(fixations_filtered_left, fixations_filtered_right, target_points)
        
        
        # RMSE values for raw and corrected data (averaged btween left- and right fixations)
        rmse_raw = (self.rmse(gaze_data_left, target_points) + self.rmse(gaze_data_right, target_points)) / 2
        rmse_cor = (self.rmse(gaze_data_left_corrected, target_points) + self.rmse(gaze_data_right_corrected, target_points)) / 2
        
        print("RMS error raw:\t\t" + str(rmse_raw))
        print("RMS error corrected:\t" + str(rmse_cor))
        print("Change:\t\t\t" + str((rmse_raw - rmse_cor) / max(rmse_raw, rmse_cor) * 100) + " %")
        
        
        
        pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
        angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
        
        pixel_err_left_corrected, pixel_err_right_corrected = self.compute_pixel_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        angle_err_left_corrected, angle_err_right_corrected = self.compute_visual_angle_error(pixel_err_left_corrected, pixel_err_right_corrected)
        
        rmse_deg_raw = (self.rmse_deg(angle_err_left) + self.rmse_deg(angle_err_right)) / 2
        rmse_deg_cor = (self.rmse_deg(angle_err_left_corrected) + self.rmse_deg(angle_err_right_corrected)) / 2
        
        print("RMS error raw (deg of visual angle):\t\t" + str(rmse_deg_raw))
        print("RMS error corrected (deg of visual angle):\t" + str(rmse_deg_cor))
        print("Change:\t\t\t" + str((rmse_deg_raw - rmse_deg_cor) / max(rmse_deg_raw, rmse_deg_cor) * 100) + " %")
        
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

        #gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        #gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)

        #self.fine_data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        #self.fine_data_correction.calibrate_left_eye(gaze_data_left_corrected)
        #self.fine_data_correction.calibrate_right_eye(gaze_data_right_corrected)


        #self.data_correction.calibrate_left_eye_seb(gaze_data_left)
        #self.data_correction.calibrate_right_eye_seb(gaze_data_right)
        
        
    def analyze_regression(self, training_filename, filtering_method = None):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, filtering_method)
        
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye_regression(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye_regression(gaze_data_right)
        #------------------------------#
        
        ### error analysis - corrected
        self.analyze_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
        ### error analysis - corrected
#        fixations_filtered_left, filtered_targets = self.reject_outliers(gaze_data_left_corrected, target_points)
#        fixations_filtered_right, filtered_targets = self.reject_outliers(gaze_data_right_corrected, target_points)
#        self.analyze_errors(fixations_filtered_left, fixations_filtered_right, target_points)
        
        
        # RMSE values for raw and corrected data (averaged btween left- and right fixations)
        rmse_raw = (self.rmse(gaze_data_left, target_points) + self.rmse(gaze_data_right, target_points)) / 2
        rmse_cor = (self.rmse(gaze_data_left_corrected, target_points) + self.rmse(gaze_data_right_corrected, target_points)) / 2
        
        print("RMS error raw:\t\t" + str(rmse_raw))
        print("RMS error corrected:\t" + str(rmse_cor))
        print("Change:\t\t\t" + str((rmse_raw - rmse_cor) / max(rmse_raw, rmse_cor) * 100) + " %")
        
        
        
        pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
        angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
        
        pixel_err_left_corrected, pixel_err_right_corrected = self.compute_pixel_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        angle_err_left_corrected, angle_err_right_corrected = self.compute_visual_angle_error(pixel_err_left_corrected, pixel_err_right_corrected)
        
        rmse_deg_raw = (self.rmse_deg(angle_err_left) + self.rmse_deg(angle_err_right)) / 2
        rmse_deg_cor = (self.rmse_deg(angle_err_left_corrected) + self.rmse_deg(angle_err_right_corrected)) / 2
        
        print("RMS error raw (deg of visual angle):\t\t" + str(rmse_deg_raw))
        print("RMS error corrected (deg of visual angle):\t" + str(rmse_deg_cor))
        print("Change:\t\t\t" + str((rmse_deg_raw - rmse_deg_cor) / max(rmse_deg_raw, rmse_deg_cor) * 100) + " %")
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
    
    def rmse(self, fixations, targets):
        fixations_filtered, filtered_targets = self.reject_outliers(fixations, targets)
        return np.sqrt(((fixations_filtered - filtered_targets) ** 2).mean())

    def rmse_deg(self, degrees):
        degrees_filtered = degrees[abs(degrees - np.mean(degrees)) < 2 * np.std(degrees)]
        return np.sqrt((degrees_filtered ** 2).mean())

    
    def reject_outliers(self, data, targets, m=1.5):
        filtered_x = [x if abs(x - np.mean(data[0,:])) < m * np.std(data[0,:]) else -1 for x in data[0,:]]
        filtered_y = [y if abs(y - np.mean(data[1,:])) < m * np.std(data[1,:]) else -1 for y in data[1,:]]
        
        filtered_data = [[],[]]
        filtered_targets = [[],[]]
        
        for x,y,tx,ty in zip(filtered_x, filtered_y, targets[0,:], targets[1,:]):
            if x >= 0.0 and y >= 0.0:
                filtered_data[0].append(x)
                filtered_data[1].append(y)
                
                filtered_targets[0].append(tx)
                filtered_targets[1].append(ty)
                
        return (np.array(filtered_data), np.array(filtered_targets))
    
    
    def reject_outliers_no_targets(self, data, m=1.5):
        return [x if abs(x - np.mean(data)) < m * np.std(data) else -1 for x in data]
    
    def analyze_errors(self, gaze_data_left, gaze_data_right, target_points):
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
        
        
        
#        self.plot_gaze_points_in_pixels(gaze_data_left, gaze_data_right, target_points, title_string="Gaze data on screen")
         
        
        
            
        
    def euclid_dist(self, a, b):
        return (a**2 + b**2) ** 0.5
    
    
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
        px_right_x = gaze_data_left[0,:] * self.screen_width_px
        px_right_y = self.screen_height_px - gaze_data_left[1,:] * self.screen_height_px
                
        
        plt.scatter(px_left_x, px_left_y, marker='x', color="red")
        # plot lines from targets to gaze points
#        for i in range(len(px_target_x)):
#            plt.plot([px_target_x[i], px_left_x[i]],[px_target_y[i], px_left_y[i]], 'k-')
        plt.xlim(0,self.screen_width_px)
        plt.ylim(0,self.screen_height_px)
        plt.title(title_string)
        plt.show()
        
        
#        plt.scatter(px_right_x, px_right_y, marker='x', color="green")
#        # plot lines from targets to gaze points
##        for i in range(len(px_target_x)):
##            plt.plot([px_target_x[i], px_right_x[i]],[px_target_y[i], px_right_y[i]], 'k-')
#        plt.xlim(0,self.screen_width_px)
#        plt.ylim(0,self.screen_height_px)
#        plt.title(title_string)
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
            
        
        px_err_left_x = self.reject_outliers_no_targets(px_err_left_x)
        px_err_left_y = self.reject_outliers_no_targets(px_err_left_y)
        px_err_right_x = self.reject_outliers_no_targets(px_err_right_x)
        px_err_right_y = self.reject_outliers_no_targets(px_err_right_y)
        
        # fit a qudratic line for the vertical errors
        poly_left_x = np.poly1d(np.polyfit(px_left_x, px_err_left_y, 2))
        poly_left_y = np.poly1d(np.polyfit(px_left_y, px_err_left_y, 2))
        poly_right_x = np.poly1d(np.polyfit(px_right_x, px_err_left_y, 2))
        poly_right_y = np.poly1d(np.polyfit(px_right_y, px_err_left_y, 2))
        
        # calculate new x's and y's
        line_y_left_x = poly_left_x(px_left_x)
        line_y_left_y = poly_left_y(px_left_y)
        line_y_right_x = poly_right_x(px_right_x)
        line_y_right_y = poly_right_y(px_right_y)
        
        plot_left_x = plt.plot(px_left_x, px_err_left_y, 'o', px_left_x, line_y_left_x)      
        plt.legend(plot_left_x, ("X Coordinate"))
        plt.xlim(0,self.screen_width_px)
        plt.title(title_string)
        plt.show()
    