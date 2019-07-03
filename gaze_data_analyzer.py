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
    
    plt.rcParams.update({'font.size': 14})

    show_graphs_bool = False
    show_rms_pixel_bool = False
    show_rms_degree_bool = False
    show_filtering = False
    show_filtering_text = False
    show_accuracy_precision_bool = True
    
    to_closest_target = False
    
    
    regression_poly_degree = 2
    
    def read_data(self, filename, remove_nan_values = True):
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
        gaze_data_left = np.transpose(np.array([eval(coord) if coord != "(nan, nan)" else (-1,-1) for coord in data_frame['left_gaze_point_on_display_area']]))
        gaze_data_right = np.transpose(np.array([eval(coord) if coord != "(nan, nan)" else (-1,-1) for coord in data_frame['right_gaze_point_on_display_area']]))
        target_points = np.transpose(np.array([eval(coord) if coord != "(nan, nan)" else (-1,-1) for coord in data_frame['current_target_point_on_display_area']]))

        return (gaze_data_left, gaze_data_right, target_points)
    
    def filtering_setup(self, gaze_data_left_temp, gaze_data_right_temp, target_points_temp, filtering_method = None, remove_outliers = True):
        
        gaze_data_left = []
        gaze_data_right = []
        target_points = []
        
        if len(gaze_data_left_temp) > 0 and len(gaze_data_right_temp) > 0 and self.show_filtering:
            self.plot_scatter(gaze_data_left_temp, gaze_data_right_temp, target_points_temp, title_string="BEFORE filtering")
            self.plot_scatter_avg(gaze_data_left_temp, gaze_data_right_temp, target_points_temp, title_string="BEFORE filtering AVG")
        
        if filtering_method == "dbscan_fixation" or filtering_method == "dbscan_pursuit":
            
            gaze_data_temp = np.mean(np.array([gaze_data_left_temp, gaze_data_right_temp]), axis=0)
            
            db_scan = dbscan.DBScan()
            
            dist_to_neighbor = 0.05
            min_size_of_cluster = 10
            if filtering_method == "dbscan_fixation":
                dist_to_neighbor = 0.01
                min_size_of_cluster = 10
                
            clusters = db_scan.run_linear(gaze_data_temp.T, dist_to_neighbor, min_size_of_cluster)
            
            
#            if self.show_graphs_bool:
#                colours = ['black', 'red', 'blue', 'cyan', 'yellow', 'purple', 'green']
#                colors = [colours[int(clusters[key]) % len(colours)] for key in clusters.keys()]
#                plt.scatter(*zip(*clusters.keys()),c=colors)
#                plt.title("DBScan", y=1.08)
#                plt.gca().xaxis.tick_top()
#                plt.xlim(0,1)
#                plt.ylim(1,0)
#                plt.show()
            
            gaze_data_left_x = []
            gaze_data_left_y = []
            gaze_data_right_x = []
            gaze_data_right_y = []
            target_points_x = []
            target_points_y = []
            
            
            for i in range(self.N):
                              
                if i < 40:
                    continue;
                
#                if i < 40 and filtering_method == "dbscan_fixation":
#                    continue;
                    
                current_target = target_points_temp[:,i]
                p = (gaze_data_temp[0, i], gaze_data_temp[1, i])
                
                if p not in clusters:
                    continue;
                    
#                current_cluster = clusters[p]
                
                # Check if current target is a new target, and if the future target is a new target

                dif_new_target_past = 50
                is_past_new_target = False
                if (i - dif_new_target_past >= 0):
                    is_past_new_target = not np.array_equal(current_target, target_points_temp[:,i-dif_new_target_past])

                dif_new_target_future = 10
                is_future_new_target = False
                if (i + dif_new_target_future < self.N):
                    is_future_new_target = not np.array_equal(current_target, target_points_temp[:,i+dif_new_target_future])
                
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
                        
            
#            if self.show_graphs_bool:
#                colours = ['black', 'red', 'blue', 'cyan', 'yellow', 'purple', 'green']
#                colors = [colours[int(clusters[key]) % len(colours)] for key in clusters.keys()]
#                plt.scatter(*zip(*clusters.keys()),c=colors)
#                plt.title("DBScan", y=1.08)
#                plt.gca().xaxis.tick_top()
#                plt.xlim(0,1)
#                plt.ylim(1,0)
#                plt.show()
            
            gaze_data_left = np.array([gaze_data_left_x, gaze_data_left_y])
            gaze_data_right = np.array([gaze_data_right_x, gaze_data_right_y])
            target_points = np.array([target_points_x, target_points_y])
            
            if self.show_filtering:
                self.plot_scatter(gaze_data_left, gaze_data_right, target_points, title_string="AFTER dbscan filter")
                self.plot_scatter_avg(gaze_data_left, gaze_data_right, target_points, title_string="AFTER dbscan AVG")

        else:
            gaze_data_left = gaze_data_left_temp
            gaze_data_right = gaze_data_right_temp
            target_points = target_points_temp
        self.N = len(target_points[0,:])
        
        return (gaze_data_left, gaze_data_right, target_points)
    
    def filtering(self, gaze_data_left_temp, gaze_data_right_temp, target_points_temp, filtering_method = None, remove_outliers = True):

        colours = ['black', 'red', 'blue', 'cyan', 'yellow', 'purple', 'green', 'brown', 'darkgrey', 'orange', 'mediumspringgreen', 'cadetblue', 'fuchsia', 'crimson']
                
        gaze_data_left = []
        gaze_data_right = []
        target_points = []
        
        before = len(target_points_temp[0,:])
        
        if len(gaze_data_left_temp) > 0 and len(gaze_data_right_temp) > 0 and self.show_filtering:
            self.plot_scatter(gaze_data_left_temp, gaze_data_right_temp, target_points_temp, title_string="BEFORE filtering")
            self.plot_scatter_avg(gaze_data_left_temp, gaze_data_right_temp, target_points_temp, title_string="BEFORE filtering AVG")

        if filtering_method == "dbscan_fixation" or filtering_method == "dbscan_pursuit":
            
            gaze_data_temp = np.mean(np.array([gaze_data_left_temp, gaze_data_right_temp]), axis=0)
            
            db_scan = dbscan.DBScan()
            
            dist_to_neighbor = 0.02
            min_size_of_cluster = 10
            if filtering_method == "dbscan_fixation":
                dist_to_neighbor = 0.01
                min_size_of_cluster = 10
                
            clusters = db_scan.run_linear(gaze_data_temp.T, dist_to_neighbor, min_size_of_cluster)
            
            
            if self.show_filtering:
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
            
            removed_gaze = 0
            for i in range(self.N):
                              
                # standard for infants 45
                # noel/gudrun tri 2p = 60
                # control 1 default =  45
                
                if i < 45 and filtering_method == "dbscan_pursuit":
                    continue;
                
                # standard for infants 45
                # standard for control 31
                # noel/gudrun 2p, 5p = 70
                # control 1 default =  45
                if i < 45 and filtering_method == "dbscan_fixation":
                    continue;
                    
                current_target = target_points_temp[:,i]
                p = (gaze_data_temp[0, i], gaze_data_temp[1, i])
                
                if p not in clusters:
                    continue;
                    
#                current_cluster = clusters[p]
                
                # Check if current target is a new target, and if the future target is a new target

                # standard 10
                # noel/gudrun 2p, 5p = 55
                
                # chrille1 default = 35
                # control 1 default =  40
                dif_new_target_past = 40
                is_past_new_target = False
                if (i - dif_new_target_past >= 0):
                    is_past_new_target = not np.array_equal(current_target, target_points_temp[:,i-dif_new_target_past])

                dif_new_target_future = 10
                is_future_new_target = False
                if (i + dif_new_target_future < self.N):
                    is_future_new_target = not np.array_equal(current_target, target_points_temp[:,i+dif_new_target_future])
                
                # For fixation filtering
                # A gaze point in a cluster has to be filtered away 
                # If the current target point has changed but the gaze point has not, the gaze point can still be in the old cluster, for the old target point
                # This gaze point should be filtered away, since it has a large visual angle error
                # If the current target is different from the previous target, but the previous target is the same as the target before that, a change has been noted
                if clusters[p] == 0 or (filtering_method == "dbscan_fixation" and (is_past_new_target or is_future_new_target)):
                    del clusters[p]
                    removed_gaze = removed_gaze + 1
                else:
                    gaze_data_left_x.append(gaze_data_left_temp[0,i])
                    gaze_data_left_y.append(gaze_data_left_temp[1,i])
                    gaze_data_right_x.append(gaze_data_right_temp[0,i])
                    gaze_data_right_y.append(gaze_data_right_temp[1,i])
                    target_points_x.append(target_points_temp[0,i])
                    target_points_y.append(target_points_temp[1,i])
                        
            
            if self.show_filtering:
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
            
            if len(gaze_data_left) > 0 and len(gaze_data_right) > 0 and self.show_filtering:
                self.plot_scatter(gaze_data_left, gaze_data_right, target_points, title_string="AFTER dbscan filter")
                self.plot_scatter_avg(gaze_data_left, gaze_data_right, target_points, title_string="AFTER dbscan AVG")


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

            if len(gaze_data_left) > 0 and len(gaze_data_right) > 0 and self.show_filtering:
                self.plot_scatter(gaze_data_left, gaze_data_right, target_points, title_string="AFTER treshold filter")
                self.plot_scatter_avg(gaze_data_left, gaze_data_right, target_points, title_string="AFTER treshold AVG")
            
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
                
                if (wait > 20):
                    gaze_data_left_x.append(gaze_data_left_temp[0,i])
                    gaze_data_left_y.append(gaze_data_left_temp[1,i])
                    gaze_data_right_x.append(gaze_data_right_temp[0,i])
                    gaze_data_right_y.append(gaze_data_right_temp[1,i])
                    target_points_x.append(target_points_temp[0,i])
                    target_points_y.append(target_points_temp[1,i])

                       
            gaze_data_left = np.array([gaze_data_left_x, gaze_data_left_y])
            gaze_data_right = np.array([gaze_data_right_x, gaze_data_right_y])
            target_points = np.array([target_points_x, target_points_y])
            
            if self.show_filtering:
                self.plot_scatter(gaze_data_left, gaze_data_right, target_points, title_string="AFTER treshold filter")
                self.plot_scatter_avg(gaze_data_left, gaze_data_right, target_points, title_string="AFTER treshold AVG")
            
        # Do nothing for filter out outliers
        elif filtering_method == None:
            gaze_data_left = gaze_data_left_temp
            gaze_data_right = gaze_data_right_temp
            target_points = target_points_temp
        
        if self.show_filtering_text:
            print("After Grace")
            print(str(before) + " - > " + str(before - removed_gaze))
            print("After DBSCAN")
            print(str(before - removed_gaze) + " - > " + str(len(target_points[0,:])))
            before_outlier = len(target_points[0,:])    
        
        if remove_outliers:
            
            pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
    
            pixel_left = [[],[]]
            pixel_right = [[],[]]
            
            for left, right in zip(pixel_err_left,pixel_err_right):
                pixel_left[0].append(left[0])
                pixel_left[1].append(left[1])
                
                pixel_right[0].append(right[0])
                pixel_right[1].append(right[1])
                
            pixel_err_left = np.array(pixel_left)
            pixel_err_right = np.array(pixel_right)
#            
#            m = 1.5
#            
#            indices_left_x = [i for i, x in enumerate(pixel_err_left[0,:]) if abs(x - np.mean(pixel_err_left[0,:])) < m * np.std(pixel_err_left[0,:])]
#            indices_left_y = [i for i, y in enumerate(pixel_err_left[1,:]) if abs(y - np.mean(pixel_err_left[1,:])) < m * np.std(pixel_err_left[1,:])]
#    
#            indices_right_x = [i for i, x in enumerate(pixel_err_right[0,:]) if abs(x - np.mean(pixel_err_right[0,:])) < m * np.std(pixel_err_right[0,:])]
#            indices_right_y = [i for i, y in enumerate(pixel_err_right[1,:]) if abs(y - np.mean(pixel_err_right[1,:])) < m * np.std(pixel_err_right[1,:])]
#    
#            indices = list(set(indices_left_x) & set(indices_left_y) & set(indices_right_x) & set(indices_right_y))
#    
#            gaze_data_left = gaze_data_left[:,indices]
#            gaze_data_right = gaze_data_right[:,indices]
#            target_points = target_points[:,indices]
#        
#            if self.show_filtering:
#                self.plot_scatter(gaze_data_left, gaze_data_right, target_points, title_string="AFTER outlier filter")
#                self.plot_scatter_avg(gaze_data_left, gaze_data_right, target_points, title_string="AFTER outlier AVG")



            errors = []
            for left,right in zip(pixel_err_left.T, pixel_err_right.T):                
                errors.append(left[0]+left[1]+right[0]+right[1])            
            
            indices_to_remove = []
            indices = np.array(range(len(errors)))
            errors = np.array(errors)
            
            #gudrun 5p linear = 0.17
            #gudrun 5p spi = 0.07

            #noel 5p linear = 0.08
            
            remove_percent = 0.10
            
            for i in range(int(len(errors)*remove_percent)):
                index = np.where(errors == np.amax(errors))[0][0]
#                print(np.amax(errors))
#                print(errors[index])
#                print(index)
#                print("")
                errors[index] = -1
                indices_to_remove.append(index)
                
            indices = np.delete(indices, indices_to_remove)
#            print(errors[indices_to_remove])
            gaze_data_left = gaze_data_left[:,indices]
            gaze_data_right = gaze_data_right[:,indices]
            target_points = target_points[:,indices]
            
            
        if self.show_filtering_text:
            print("After Outlier")
            print(str(before_outlier) + " - > " + str(len(target_points[0,:])))
            print((float(before_outlier)-float(len(target_points[0,:])))/float(before)*100.0)
        
        self.N = len(target_points[0,:])
        
        return (gaze_data_left, gaze_data_right, target_points)
    
    def center_by_cluster(self, gaze_data_left, gaze_data_right):
        return (self.data_correction.adjust_by_cluster_center(gaze_data_left), self.data_correction.adjust_by_cluster_center(gaze_data_right))
    
    def animate(self, training_filename):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename, remove_nan_values = False)
        
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
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
         
        # filter gaze points, remove outliers
        # then redefine target as those closest to gaze points
        
        if self.to_closest_target:
            target_points = self.find_closest_target(target_points, gaze_data_left, gaze_data_right)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye(gaze_data_left)
        self.data_correction.calibrate_right_eye(gaze_data_right)
        
    
    def analyze(self, training_filename, filtering_method = None, output = "points", remove_outliers=True, mean_cluster_replacement=False):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
        
        if len(gaze_data_left) == 0 and len(gaze_data_right) == 0:
            return None
        
        if self.to_closest_target:
            target_points = self.find_closest_target(target_points, gaze_data_left, gaze_data_right)
        
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
#        MEAN OF CLUSTERING ON/OFF
#        gaze_data_left_corrected, gaze_data_right_corrected = self.center_by_cluster(gaze_data_left, gaze_data_right)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#        gaze_data_left_corrected = gaze_data_left
#        gaze_data_right_corrected = gaze_data_right
        
#        MEAN OF CLUSTERING ON/OFF
#        if mean_cluster_replacement == True:
#            gaze_data_left_corrected, gaze_data_right_corrected = self.center_by_cluster(gaze_data_left_corrected, gaze_data_right_corrected)
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
        
        
        angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected, angle_avg, angle_avg_corrected = self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        # formula proof accuracy (angular offset) calc
#        angle_left, angle_right, angle_avg = self.compute_angular_offset(gaze_data_left, gaze_data_right, target_points)
#        angle_left_corrected, angle_right_corrected, angle_avg_corrected = self.compute_angular_offset(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
#        
#        accuracy_raw = np.mean(angle_avg)
#        accuracy_corrected = np.mean(angle_avg_corrected)
#        
#        print("")
#        print("############################################")
#        print("Accuracy: "+u"\u03B8"+ "_offset")
#        print("Accuracy (raw)\t\t" + str(accuracy_raw))
#        print("Accuracy (corrected)\t" + str(accuracy_corrected))
#        print("-----------")
#        print("Change\t\t\t" + str((accuracy_raw - accuracy_corrected) / max(accuracy_raw, accuracy_corrected) * 100) + " %")
#        print("############################################")
#        
#        
#        # formula proof precision calc
#        precision_avg = (np.mean([theta**2 for theta in angle_avg]))**0.5
#        precision_avg_corrected = (np.mean([theta**2 for theta in angle_avg_corrected]))**0.5
#        
#        print("")
#        print("############################################")
#        print("Precision: RMS("+u"\u03B8" + ")")
#        print("Precision (raw)\t\t" + str(precision_avg))
#        print("Precision (corrected)\t" + str(precision_avg_corrected))
#        print("-----------")
#        print("Change\t\t\t" + str((precision_avg - precision_avg_corrected) / max(precision_avg, precision_avg_corrected) * 100) + " %")
#        print("############################################")
        
        
        if output == "values":
            return (angle_avg, angle_avg_corrected)
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
    
    # set up the transformation matrices 
    def setup_coef(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
         
        # filter gaze points, remove outliers
        # then redefine target as those closest to gaze points
        
        if self.to_closest_target:
            target_points = self.find_closest_target(target_points, gaze_data_left, gaze_data_right)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye_coef(gaze_data_left)
        self.data_correction.calibrate_right_eye_coef(gaze_data_right)
        
    
    def analyze_coef(self, training_filename, filtering_method = None, output = "points", remove_outliers=True, mean_cluster_replacement=False):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
        
        if len(gaze_data_left) == 0 and len(gaze_data_right) == 0:
            return None
        
        if self.to_closest_target:
            target_points = self.find_closest_target(target_points, gaze_data_left, gaze_data_right)
        
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
#        MEAN OF CLUSTERING ON/OFF
#        gaze_data_left_corrected, gaze_data_right_corrected = self.center_by_cluster(gaze_data_left, gaze_data_right)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye_coef(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye_coef(gaze_data_right)
#        gaze_data_left_corrected = gaze_data_left
#        gaze_data_right_corrected = gaze_data_right
        
#        MEAN OF CLUSTERING ON/OFF
#        if mean_cluster_replacement == True:
#            gaze_data_left_corrected, gaze_data_right_corrected = self.center_by_cluster(gaze_data_left_corrected, gaze_data_right_corrected)
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
        
        
        angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected, angle_avg, angle_avg_corrected = self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        # formula proof accuracy (angular offset) calc
#        angle_left, angle_right, angle_avg = self.compute_angular_offset(gaze_data_left, gaze_data_right, target_points)
#        angle_left_corrected, angle_right_corrected, angle_avg_corrected = self.compute_angular_offset(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
#        
#        accuracy_raw = np.mean(angle_avg)
#        accuracy_corrected = np.mean(angle_avg_corrected)
#        
#        print("")
#        print("############################################")
#        print("Accuracy: "+u"\u03B8"+ "_offset")
#        print("Accuracy (raw)\t\t" + str(accuracy_raw))
#        print("Accuracy (corrected)\t" + str(accuracy_corrected))
#        print("-----------")
#        print("Change\t\t\t" + str((accuracy_raw - accuracy_corrected) / max(accuracy_raw, accuracy_corrected) * 100) + " %")
#        print("############################################")
#        
#        
#        # formula proof precision calc
#        precision_avg = (np.mean([theta**2 for theta in angle_avg]))**0.5
#        precision_avg_corrected = (np.mean([theta**2 for theta in angle_avg_corrected]))**0.5
#        
#        print("")
#        print("############################################")
#        print("Precision: RMS("+u"\u03B8" + ")")
#        print("Precision (raw)\t\t" + str(precision_avg))
#        print("Precision (corrected)\t" + str(precision_avg_corrected))
#        print("-----------")
#        print("Change\t\t\t" + str((precision_avg - precision_avg_corrected) / max(precision_avg, precision_avg_corrected) * 100) + " %")
#        print("############################################")
        
        
        if output == "values":
            return (angle_avg, angle_avg_corrected)
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
    
    def getTransformationLeft(self):
        return self.data_correction.transformation_matrix_left_eye
    
    def getTransformationRight(self):
        return self.data_correction.transformation_matrix_right_eye
    
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
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = True)
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
        print("With a given optimization of " + str(best_optimize))
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
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye_poly(gaze_data_left)
        self.data_correction.calibrate_right_eye_poly(gaze_data_right)
        
        
        
    def analyze_poly(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
        
        
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
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
        if output == "values":
            return (rmse_deg_raw, rmse_deg_cor, rmse_deg_imp)
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
    
   
        
    
     # set up the transformation matrices 
    def setup_regression(self, config_file, cal_filename, filtering_method = None, poly_degree=2):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        self.regression_poly_degree = poly_degree
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_eyes_regression(gaze_data_left, gaze_data_right, degree=poly_degree)
        
        
    def analyze_regression(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)

        
        
        
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
        
        
        m = max(gaze_data_right[0,:])
        maxindex = [i for i, j in enumerate(gaze_data_right[0,:]) if j == m]
#        print("Before Correction:\t" + str((gaze_data_right[0,maxindex]*self.screen_width_px, self.screen_height_px - gaze_data_right[1,maxindex]*self.screen_height_px)))
#        print("After Correction:\t" + str((gaze_data_right_corrected[0,maxindex]*self.screen_width_px, self.screen_height_px - gaze_data_right_corrected[1,maxindex]*self.screen_height_px)))
#        print(self.data_correction.poly_right_x)
#        print(self.data_correction.poly_right_y)
#        print(self.data_correction.poly_right_x(gaze_data_right[0,maxindex])*self.screen_height_px)
#        print(self.data_correction.poly_right_y(gaze_data_right[1,maxindex])*self.screen_width_px)
#        print("")
        
        # RMSE values for raw and corrected data (averaged btween left- and right fixations)
        rmse_raw, rmse_cor, rmse_imp = self.show_rms_pixel(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)                
        
#        pixel_err_left, pixel_err_right = self.compute_pixel_errors_to_closest_target(gaze_data_left, gaze_data_right, target_points)
        pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
        angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
        
        pixel_err_left_corrected, pixel_err_right_corrected = self.compute_pixel_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        angle_err_left_corrected, angle_err_right_corrected = self.compute_visual_angle_error(pixel_err_left_corrected, pixel_err_right_corrected)
        
        rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = self.show_rms_degree(angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
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
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye_seb(gaze_data_left)
        self.data_correction.calibrate_right_eye_seb(gaze_data_right)

#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#
#        self.data_correction.calibrate_left_eye_seb(gaze_data_left_corrected)
#        self.data_correction.calibrate_right_eye_seb(gaze_data_right_corrected)
            
        
        
    def analyze_seb(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
 
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
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
        if output == "values":
            return (rmse_deg_raw, rmse_deg_cor, rmse_deg_imp)
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
    
    # set up the transformation matrices 
    def setup_translate(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.affine_left_eye(gaze_data_left)
        self.data_correction.affine_right_eye(gaze_data_right)
            
        
        
    def analyze_translate(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
 
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.affine_adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.affine_adjust_right_eye(gaze_data_right)
        
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
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
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
    def setup_translate_mix(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.affine_calibrate_left_eye_seb(gaze_data_left)
        self.data_correction.affine_calibrate_right_eye_seb(gaze_data_right)
        
    def analyze_translate_mix(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
 
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.affine_adjust_left_eye_seb_2(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.affine_adjust_right_eye_seb_2(gaze_data_right)
        
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
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
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
    def setup_affine2(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        if self.to_closest_target:
            target_points = self.find_closest_target(target_points, gaze_data_left, gaze_data_right)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.affine_left_eye2(gaze_data_left)
        self.data_correction.affine_right_eye2(gaze_data_right)

#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#
#        self.data_correction.affine_left_eye(gaze_data_left_corrected)
#        self.data_correction.affine_right_eye(gaze_data_right_corrected)

#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#
#        self.data_correction.calibrate_left_eye_seb(gaze_data_left_corrected)
#        self.data_correction.calibrate_right_eye_seb(gaze_data_right_corrected)
            
        
        
    def analyze_affine2(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
 
        if self.to_closest_target:
            target_points = self.find_closest_target(target_points, gaze_data_left, gaze_data_right)
    
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.affine_adjust_left_eye2(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.affine_adjust_right_eye2(gaze_data_right)
        
#        gaze_data_left_corrected, gaze_data_right_corrected = self.center_by_cluster(gaze_data_left_corrected, gaze_data_right_corrected)
        
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
#        rmse_raw, rmse_cor, rmse_imp = self.show_rms_pixel(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)                
#        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
#        pixel_err_left, pixel_err_right = self.compute_pixel_errors_to_closest_target(gaze_data_left, gaze_data_right, target_points)
        pixel_err_left, pixel_err_right = self.compute_pixel_errors(gaze_data_left, gaze_data_right, target_points)
        angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
        
        pixel_err_left_corrected, pixel_err_right_corrected = self.compute_pixel_errors(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        angle_err_left_corrected, angle_err_right_corrected = self.compute_visual_angle_error(pixel_err_left_corrected, pixel_err_right_corrected)
        
#        rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = self.show_rms_degree(angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
        angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected, angle_avg, angle_avg_corrected = self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
        if output == "values":
            return (angle_avg, angle_avg_corrected)
        
        return (target_points, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected)
        
    
    # set up the transformation matrices 
    def setup_affine_mix(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.affine_left_eye2_mix(gaze_data_left)
        self.data_correction.affine_right_eye2_mix(gaze_data_right)

#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#
#        self.data_correction.affine_left_eye(gaze_data_left_corrected)
#        self.data_correction.affine_right_eye(gaze_data_right_corrected)

#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#
#        self.data_correction.calibrate_left_eye_seb(gaze_data_left_corrected)
#        self.data_correction.calibrate_right_eye_seb(gaze_data_right_corrected)
            
        
        
    def analyze_affine_mix(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
 
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.affine_adjust_left_eye2_mix(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.affine_adjust_right_eye2_mix(gaze_data_right)
        
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
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
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
    def setup_affine(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.affine_left_eye(gaze_data_left)
        self.data_correction.affine_right_eye(gaze_data_right)

        gaze_data_left_corrected = self.data_correction.affine_adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.affine_adjust_right_eye(gaze_data_right)

        self.data_correction.calibrate_left_eye(gaze_data_left_corrected)
        self.data_correction.calibrate_right_eye(gaze_data_right_corrected)

#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#
#        self.data_correction.calibrate_left_eye_seb(gaze_data_left_corrected)
#        self.data_correction.calibrate_right_eye_seb(gaze_data_right_corrected)
            
        
        
    def analyze_affine(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
 
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        affine_gaze_data_left_corrected = self.data_correction.affine_adjust_left_eye(gaze_data_left)
        affine_gaze_data_right_corrected = self.data_correction.affine_adjust_right_eye(gaze_data_right)
        gaze_data_left_corrected = self.data_correction.adjust_left_eye(affine_gaze_data_left_corrected)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(affine_gaze_data_right_corrected)
        
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
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
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
    def setup_affine_weighted(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.affine_calibrate_left_eye_seb(gaze_data_left)
        self.data_correction.affine_calibrate_right_eye_seb(gaze_data_right)

        gaze_data_left_corrected = self.data_correction.affine_adjust_left_eye_seb_2(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.affine_adjust_right_eye_seb_2(gaze_data_right)

        self.data_correction.calibrate_left_eye(gaze_data_left_corrected)
        self.data_correction.calibrate_right_eye(gaze_data_right_corrected)
#        
#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#
#        self.data_correction.calibrate_left_eye_seb(gaze_data_left_corrected)
#        self.data_correction.calibrate_right_eye_seb(gaze_data_right_corrected)
            
        
        
    def analyze_affine_weighted(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
 
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.affine_adjust_left_eye_seb_2(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.affine_adjust_right_eye_seb_2(gaze_data_right)
        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left_corrected)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right_corrected)
        
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
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
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
    def setup_affine_revert(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye(gaze_data_left)
        self.data_correction.calibrate_right_eye(gaze_data_right)

        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)

        self.data_correction.affine_left_eye(gaze_data_left_corrected)
        self.data_correction.affine_right_eye(gaze_data_right_corrected)



#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#
#        self.data_correction.calibrate_left_eye_seb(gaze_data_left_corrected)
#        self.data_correction.calibrate_right_eye_seb(gaze_data_right_corrected)
            
        
        
    def analyze_affine_revert(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
 
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
        gaze_data_left_corrected = self.data_correction.affine_adjust_left_eye(gaze_data_left_corrected)
        gaze_data_right_corrected = self.data_correction.affine_adjust_right_eye(gaze_data_right_corrected)
        
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
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
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
    def setup_affine_revert_weighted(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye(gaze_data_left)
        self.data_correction.calibrate_right_eye(gaze_data_right)

        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)

        self.data_correction.affine_calibrate_left_eye_seb(gaze_data_left_corrected)
        self.data_correction.affine_calibrate_right_eye_seb(gaze_data_right_corrected)



#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#
#        self.data_correction.calibrate_left_eye_seb(gaze_data_left_corrected)
#        self.data_correction.calibrate_right_eye_seb(gaze_data_right_corrected)
            
        
        
    def analyze_affine_revert_weighted(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
 
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
        gaze_data_left_corrected = self.data_correction.affine_adjust_left_eye_seb_2(gaze_data_left_corrected)
        gaze_data_right_corrected = self.data_correction.affine_adjust_right_eye_seb_2(gaze_data_right_corrected)
        
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
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
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
    def setup_affine_poly(self, config_file, cal_filename, filtering_method = None):
        
        # read config csv file
        data_frame = pd.read_csv(config_file, delimiter=";")
        
        # read global config variables in
        self.screen_width_px = data_frame['Screen width (px)'][0]
        self.screen_height_px = data_frame['Screen height (px)'][0]
        self.screen_size_diag_inches = data_frame['Screen size (inches)'][0]
        self.dist_to_screen_cm = data_frame['Distance to screen (cm)'][0]
        self.ppcm = math.sqrt(self.screen_width_px**2 + self.screen_height_px**2) / (self.screen_size_diag_inches*2.54)
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_eyes_regression(gaze_data_left, gaze_data_right)

        gaze_data_left_corrected = self.data_correction.adjust_left_eye_regression(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye_regression(gaze_data_right)

        self.data_correction.calibrate_left_eye(gaze_data_left_corrected)
        self.data_correction.calibrate_right_eye(gaze_data_right_corrected)

#        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
#        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)
#
#        self.data_correction.calibrate_left_eye_seb(gaze_data_left_corrected)
#        self.data_correction.calibrate_right_eye_seb(gaze_data_right_corrected)
            
        
        
    def analyze_affine_poly(self, training_filename, filtering_method = None, output = "points", remove_outliers=True):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = remove_outliers)
 
        ### error analysis - raw
        self.analyze_errors(gaze_data_left, gaze_data_right, target_points)
        
        #------ correct raw data ------#
        affine_gaze_data_left_corrected = self.data_correction.adjust_left_eye_regression(gaze_data_left)
        affine_gaze_data_right_corrected = self.data_correction.adjust_right_eye_regression(gaze_data_right)
        gaze_data_left_corrected = self.data_correction.adjust_left_eye(affine_gaze_data_left_corrected)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(affine_gaze_data_right_corrected)
        
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
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
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
        
        gaze_data_left, gaze_data_right, target_points = self.read_data(cal_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering_setup(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = False)
        
        self.data_correction = dc.DataCorrection(target_points, self.screen_width_px, self.screen_height_px)
        self.data_correction.calibrate_left_eye(gaze_data_left)
        self.data_correction.calibrate_right_eye(gaze_data_right)

        gaze_data_left_corrected = self.data_correction.adjust_left_eye(gaze_data_left)
        gaze_data_right_corrected = self.data_correction.adjust_right_eye(gaze_data_right)

        self.data_correction.calibrate_left_eye_poly(gaze_data_left_corrected)
        self.data_correction.calibrate_right_eye_poly(gaze_data_right_corrected)
            
        
    def analyze_two_layer(self, training_filename, filtering_method = None):
        gaze_data_left, gaze_data_right, target_points = self.read_data(training_filename)
        gaze_data_left, gaze_data_right, target_points = self.filtering(gaze_data_left, gaze_data_right, target_points, filtering_method, remove_outliers = True)

        
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
        self.show_accuracy_precision(gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points)
        
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
    
    def show_accuracy_precision(self, gaze_data_left, gaze_data_right, gaze_data_left_corrected, gaze_data_right_corrected, target_points):

        # formula proof accuracy (angular offset) calc
        angle_left, angle_right, angle_avg = self.compute_angular_offset(gaze_data_left, gaze_data_right, target_points)
        angle_left_corrected, angle_right_corrected, angle_avg_corrected = self.compute_angular_offset(gaze_data_left_corrected, gaze_data_right_corrected, target_points)
                
        accuracy_raw = np.mean(angle_avg)
        accuracy_corrected = np.mean(angle_avg_corrected)
        
        # formula proof precision calc
        precision_avg = (np.mean([theta**2 for theta in angle_avg]))**0.5
        precision_avg_corrected = (np.mean([theta**2 for theta in angle_avg_corrected]))**0.5
            
        if self.show_accuracy_precision_bool:
            
            print("")
            print("############################################")
            print("Accuracy: "+u"\u03B8"+ "_offset")
            print("Accuracy (raw)\t\t" + str(accuracy_raw))
            print("Accuracy (corrected)\t" + str(accuracy_corrected))
            print("-----------")
            print("Change\t\t\t" + str((accuracy_raw - accuracy_corrected) / max(accuracy_raw, accuracy_corrected) * 100) + " %")
            print("############################################")
            
            
            
#            print("")
#            print("############################################")
#            print("Precision: RMS("+u"\u03B8" + ")")
#            print("Precision (raw)\t\t" + str(precision_avg))
#            print("Precision (corrected)\t" + str(precision_avg_corrected))
#            print("-----------")
#            print("Change\t\t\t" + str((precision_avg - precision_avg_corrected) / max(precision_avg, precision_avg_corrected) * 100) + " %")
#            print("############################################")
    
        return angle_left, angle_right, angle_left_corrected, angle_right_corrected, angle_avg, angle_avg_corrected
    
    
    def rmse(self, fixations, targets):
        return np.sqrt(((fixations - targets) ** 2).mean())
#        fixations_filtered, filtered_targets = self.reject_outliers(fixations, targets)
#        return np.sqrt(((fixations_filtered - filtered_targets) ** 2).mean())

    def rmse_deg(self, degrees):
        return np.sqrt((degrees ** 2).mean())
#        degrees_filtered = degrees[abs(degrees - np.mean(degrees)) < 2 * np.std(degrees)]
#        return np.sqrt((degrees_filtered ** 2).mean())
    
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
#            angle_err_left, angle_err_right = self.compute_visual_angle_error(pixel_err_left, pixel_err_right)
            angle_err_left, angle_err_right, angle_err_avg = self.compute_angular_offset(gaze_data_left, gaze_data_right, target_points)
            
            
            
                    #Scatter plot for fixations
            self.plot_scatter(gaze_data_left, gaze_data_right, target_points, title_string="")
#            self.plot_pixel_errors(pixel_dist_err_left, pixel_dist_err_right, title_string="Pixel distance error")
            self.plot_pixel_errors(pixel_dist_err_left, pixel_dist_err_right, title_string="")
            self.plot_angle_errors(angle_err_left, angle_err_right, title_string="Visual angle error")
#            self.plot_gaze_points_in_pixels(gaze_data_left, gaze_data_right, target_points, title_string="Gaze data on screen", poly_degree=self.regression_poly_degree)            
        
            
        
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
    
    
    def compute_angular_offset(self, gaze_data_left, gaze_data_right, target_points):
        
        visual_angle_left = []
        visual_angle_right = []
        visual_angle_avg = []
        
        for gazepoint_left_x, gazepoint_left_y, gazepoint_right_x, gazepoint_right_y, targetpoint_x, targetpoint_y in zip(gaze_data_left[0,:], gaze_data_left[1,:], gaze_data_right[0,:], gaze_data_right[1,:], target_points[0,:], target_points[1,:]):
            
            # convert normalized coordinates to pixel coordinates (as on screen)
            gazepoint_left_x *= self.screen_width_px
            gazepoint_left_y *= self.screen_height_px
            gazepoint_right_x *= self.screen_width_px
            gazepoint_right_y *= self.screen_height_px
            targetpoint_x *= self.screen_width_px
            targetpoint_y *= self.screen_height_px
            
            
            theta_left = math.atan(((abs(targetpoint_x-gazepoint_left_x)**2 + abs(targetpoint_y-gazepoint_left_y)**2))**0.5/(self.dist_to_screen_cm*self.ppcm)) * 180 / np.pi
            theta_right = math.atan(((abs(targetpoint_x-gazepoint_right_x)**2 + abs(targetpoint_y-gazepoint_right_y)**2))**0.5/(self.dist_to_screen_cm*self.ppcm)) * 180 / np.pi
            theta_avg = (theta_left + theta_right) / 2
            
            visual_angle_left.append(theta_left)
            visual_angle_right.append(theta_right)
            visual_angle_avg.append(theta_avg)
            
#        print("AVG ANGULAR OFFSET: " + str(np.mean(visual_angle_avg)))
        return visual_angle_left, visual_angle_right, visual_angle_avg
    
    
    def compute_precision(self, gaze_data_left, gaze_data_right, target_points):
        
        precision_left = []
        precision_right = []
        precision_avg = []
        
        for gazepoint_left_x, gazepoint_left_y, gazepoint_right_x, gazepoint_right_y, targetpoint_x, targetpoint_y in zip(gaze_data_left[0,:], gaze_data_left[1,:], gaze_data_right[0,:], gaze_data_right[1,:], target_points[0,:], target_points[1,:]):
            
            # convert normalized coordinates to pixel coordinates (as on screen)
            gazepoint_left_x *= self.screen_width_px
            gazepoint_left_y *= self.screen_height_px
            gazepoint_right_x *= self.screen_width_px
            gazepoint_right_y *= self.screen_height_px
            targetpoint_x *= self.screen_width_px
            targetpoint_y *= self.screen_height_px
            
            
            theta_left = math.atan(((abs(targetpoint_x-gazepoint_left_x)**2 + abs(targetpoint_y-gazepoint_left_y)**2))**0.5/(self.dist_to_screen_cm*self.ppcm)) * 180 / np.pi
            theta_right = math.atan(((abs(targetpoint_x-gazepoint_right_x)**2 + abs(targetpoint_y-gazepoint_right_y)**2))**0.5/(self.dist_to_screen_cm*self.ppcm)) * 180 / np.pi
            theta_avg = (theta_left + theta_right) / 2
            
            visual_angle_left.append(theta_left)
            visual_angle_right.append(theta_right)
            visual_angle_avg.append(theta_avg)
            
        return visual_angle_left, visual_angle_right
    
          
  
    def plot_scatter(self, gaze_data_left, gaze_data_right, targets, title_string="", show=True):
        x_left = gaze_data_left[0,:]
        y_left = gaze_data_left[1,:]
        x_right = gaze_data_right[0,:]
        y_right = gaze_data_right[1,:]
        x_targets = targets[0,:]
        y_targets = targets[1,:]
        
        
        
    
        fig = plt.figure()
        ax = fig.gca()
        
#        stimuli1 = plt.Circle((0.25,0.25), 0.0375, fill=True, color='#1f77b4', linewidth=3, label="stimuli")
#        stimuli2 = plt.Circle((0.25,0.75), 0.0375, fill=True, color='#1f77b4', linewidth=3, label="stimuli")
#        stimuli3 = plt.Circle((0.75,0.25), 0.0375, fill=True, color='#1f77b4', linewidth=3, label="stimuli")
#        stimuli4 = plt.Circle((0.75,0.75), 0.0375, fill=True, color='#1f77b4', linewidth=3, label="stimuli")
#        stimuli5 = plt.Circle((0.5,0.5), 0.0375, fill=True, color='#1f77b4', linewidth=3, label="stimuli")
#        ax.add_artist(stimuli1)
#        ax.add_artist(stimuli2)
#        ax.add_artist(stimuli3)
#        ax.add_artist(stimuli4)
#        ax.add_artist(stimuli5)
                
        scatter_left = plt.scatter(x_left, y_left, marker='x', color='red',zorder=2)
        scatter_right = plt.scatter(x_right, y_right, marker='x', color='green',zorder=2)
        scatter_target = plt.scatter(x_targets, y_targets, marker='o', color='black',zorder=2)
        
        plt.legend((scatter_left, scatter_right),("left eye", "right eye"))
        
        ax.legend((scatter_left, scatter_right, scatter_target),
                   ("left eye", "right eye", "target points"), loc=0)
        ax.set_title(title_string, y=1.08)
        ax.xaxis.tick_top()
        ax.set_xlim(0,1)
        ax.set_ylim(1,0)
#        plt.xlim(0.65,0.9)
#        plt.ylim(0.7,0.8)
        if (show == True):
            plt.show()
        
    def plot_scatter_avg(self, gaze_data_left, gaze_data_right, targets, title_string=""):
        gaze_data_avg = np.mean(np.array([gaze_data_left, gaze_data_right]), axis=0)
        
        scatter_avg = plt.scatter(gaze_data_avg[0,:], gaze_data_avg[1,:], marker='x', color='blue')
        scatter_target = plt.scatter(targets[0,:], targets[1,:], marker='^', color='black')
            
        plt.legend((scatter_avg, scatter_target),
                   ("avg eye", "target points"))
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
        plt.xlabel("Observation")
        plt.ylabel("Normalized pixel error")
        plt.title(title_string)
        plt.ylim(0,y_max)
        plt.show()
        
    def plot_gaze_points_in_pixels(self, gaze_data_left, gaze_data_right, target_points, title_string="", poly_degree=2):
        # the gaze data recorded is normalized
        # flip y-coordinates to turn recording coordinate system (origo in top-left) into screen coordinate system (origo in bottom-left)
        px_left_x = gaze_data_left[0,:] * self.screen_width_px
        px_left_y = self.screen_height_px - gaze_data_left[1,:] * self.screen_height_px
        px_right_x = gaze_data_right[0,:] * self.screen_width_px
        px_right_y = self.screen_height_px - gaze_data_right[1,:] * self.screen_height_px
        
        px_target_x = target_points[0,:] * self.screen_width_px
        px_target_y = self.screen_height_px - target_points[1,:] * self.screen_height_px
        
        
        gaze_data_avg = np.mean(np.array([gaze_data_left, gaze_data_right]), axis=0)
        
        px_gaze_avg_x = gaze_data_avg[0,:] * self.screen_width_px
        px_gaze_avg_y = self.screen_height_px - gaze_data_avg[1,:] * self.screen_height_px
        
#        px_left = np.array([px_left_x, px_left_y])
#        px_right = np.array([px_right_x, px_right_y])        
#        px_avg = (px_left + px_right)/2
        
        
        
        fig = plt.figure(1, figsize=(18,12))
        subplot_gaze_pixels = fig.add_subplot(2,2,2)
        subplot_vertical_err = fig.add_subplot(2,2,4)
        subplot_horizontal_err = fig.add_subplot(2,2,1)
        
        ## PLOT LEFT EYE AND TARGETS
#        scatter_left = plt.scatter(px_left_x, px_left_y, marker='x', color="red")
#        scatter_targets = plt.scatter(px_target_x, px_target_y, marker='o', color="black")
#        # plot lines from targets to gaze points
##        for i in range(len(px_target_x)):
##            plt.plot([px_target_x[i], px_left_x[i]],[px_target_y[i], px_left_y[i]], 'k-')
#        plt.xlim(0,self.screen_width_px)
#        plt.ylim(0,self.screen_height_px)
#        plt.title("Left eye gaze data as on screen")
#        plt.xlabel("Screen width (pixels)")
#        plt.ylabel("Screen height (pixels)")
#        plt.legend((scatter_left, scatter_targets),
#                   ("left eye", "target points"))
#        plt.show()
        
        ## PLOT RIGHT EYE AND TARGETS
        scatter_right = subplot_gaze_pixels.scatter(px_right_x, px_right_y, marker='x', color="green")
        scatter_targets = subplot_gaze_pixels.scatter(px_target_x, px_target_y, marker='o', color="black")
        # plot lines from targets to gaze points
#        for i in range(len(px_target_x)):
#            plt.plot([px_target_x[i], px_left_x[i]],[px_target_y[i], px_left_y[i]], 'k-')
        subplot_gaze_pixels.set_xlim(0,self.screen_width_px)
        subplot_gaze_pixels.set_ylim(0,self.screen_height_px)
#        subplot_gaze_pixels.set_title("Right eye gaze data as on screen")
        subplot_gaze_pixels.set_xlabel("Screen width (pixels)", fontsize=18)
        subplot_gaze_pixels.set_ylabel("Screen height (pixels)", fontsize=18)
        subplot_gaze_pixels.legend((scatter_right, scatter_targets),
                   ("right eye", "target points"))
##        plt.show()
        
        subplot_gaze_pixels.tick_params(axis='both', which='major', labelsize=14)
        
        
        
        
#        scatter_avg = subplot_gaze_pixels.scatter(px_gaze_avg_x, px_gaze_avg_y, marker='x')
#        scatter_targets = subplot_gaze_pixels.scatter(px_target_x, px_target_y, marker='o', color="black")
#        
#        subplot_gaze_pixels.set_xlim(0,self.screen_width_px)
#        subplot_gaze_pixels.set_ylim(0,self.screen_height_px)
##        subplot_gaze_pixels.set_title("Avg eye gaze data as on screen")
#        subplot_gaze_pixels.set_xlabel("Screen width (pixels)")
#        subplot_gaze_pixels.set_ylabel("Screen height (pixels)")
#        subplot_gaze_pixels.legend((scatter_avg, scatter_targets), ("Gaze points (avg)", "target points"))
        
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
            
        px_err_avg_x = np.mean(np.array([px_err_left_x, px_err_right_x]), axis=0)
        px_err_avg_y = np.mean(np.array([px_err_left_y, px_err_right_y]), axis=0)
            
#        px_err_left_x, px_left_y = self.data_correction.reject_outliers(px_err_left_x, px_left_y)
#        px_err_left_y, px_left_x = self.data_correction.reject_outliers(px_err_left_y, px_left_x)
#        px_err_right_x, px_right_y = self.data_correction.reject_outliers(px_err_right_x, px_right_y)
#        px_err_right_y, px_right_x = self.data_correction.reject_outliers(px_err_right_y, px_right_x)
        
        # fit a qudratic line for the vertical errors
#        poly_left_x = np.poly1d(np.polyfit(px_left_x, px_err_left_y, 2))
#        poly_left_y = np.poly1d(np.polyfit(px_left_y, px_err_left_x, 2))
#        poly_right_x = np.poly1d(np.polyfit(px_right_x, px_err_right_y, 2))
#        poly_right_y = np.poly1d(np.polyfit(px_right_y, px_err_right_x, 2))
        
        
        poly_left_x = np.poly1d(np.polyfit(px_left_x, px_err_left_y, poly_degree))
        poly_left_y = np.poly1d(np.polyfit(px_left_y, px_err_left_x, poly_degree))
        poly_right_x = np.poly1d(np.polyfit(px_right_x, px_err_right_y, poly_degree))
        poly_right_y = np.poly1d(np.polyfit(px_right_y, px_err_right_x, poly_degree))
#        
#        poly_right_x, det_right_x = self.polyfit(px_right_x, px_err_right_y, poly_degree)
#        poly_right_y, det_right_y = self.polyfit(px_right_y, px_err_right_x, poly_degree)
        
        poly_avg_x = np.poly1d(np.polyfit(px_gaze_avg_x, px_err_avg_y, poly_degree))
        poly_avg_y = np.poly1d(np.polyfit(px_gaze_avg_y, px_err_avg_x, poly_degree))
        
        
        
        
        # calculate new x's and y's
#        line_y_left_x = poly_left_x(px_left_x)
#        line_y_left_y = poly_left_y(px_left_y)
#        line_y_right_x = poly_right_x(px_right_x)
#        line_y_right_y = poly_right_y(px_right_y)
        
#        
#        subplot_vertical_err.scatter(px_right_x, [-a for a in px_err_right_y])  
#        px_right_x_sorted = np.sort(px_right_x)
#        subplot_vertical_err.plot(px_right_x_sorted, -poly_right_x(px_right_x_sorted), color="orange") 
##        subplot_vertical_err.set_title("Right eye vertical error as gaze varies horizontally")
#        subplot_vertical_err.set_xlabel("Screen width (pixels)")
#        subplot_vertical_err.set_ylabel("Vertical error (pixels)")
#        subplot_vertical_err.set_xlim(0, self.screen_width_px)
#        
#        
#         
#        subplot_horizontal_err.scatter(px_err_right_x, px_right_y)  
#        px_right_y_sorted = np.sort(px_right_y)
#        subplot_horizontal_err.plot(poly_right_y(px_right_y_sorted), px_right_y_sorted, color="orange")
##        subplot_horizontal_err.set_title("Right eye horizontal error as gaze varies vertically")
#        subplot_horizontal_err.set_xlabel("Horizontal error (pixels)")
#        subplot_horizontal_err.set_ylabel("Screen height (pixels)")
#        subplot_horizontal_err.set_ylim(0,self.screen_height_px)
##        
#        plt.show()
        
        fitname = "linear" if poly_degree == 1 else "quadratic" if poly_degree == 2 else "cubic" if poly_degree == 3 else "quartic" if poly_degree == 4 else "quintic" if poly_degree == 5 else "sextic" if poly_degree == 6 else "7+"
        
        
        subplot_vertical_err.scatter(px_right_x, [-a for a in px_err_right_y])  
        gaze_data_right_x = np.sort(gaze_data_right[0,:])
        subplot_vertical_err.plot(gaze_data_right_x*self.screen_width_px, self.data_correction.poly_right_x(gaze_data_right_x)*self.screen_height_px, color="orange", linewidth=3.0) 
#        subplot_vertical_err.set_title("Right eye vertical error as gaze varies horizontally")
        subplot_vertical_err.set_xlabel("Screen width (pixels)", fontsize=18)
        subplot_vertical_err.set_ylabel("Vertical error (pixels)", fontsize=18)
        subplot_vertical_err.set_xlim(0, self.screen_width_px)
        subplot_vertical_err.set_ylim(-40, 100)
        subplot_vertical_err.tick_params(axis='both', which='major', labelsize=14)
        subplot_vertical_err.legend([fitname + " fit","gaze error (right eye)"])
         
        subplot_horizontal_err.scatter(px_err_right_x, px_right_y)  
        gaze_data_right_y = np.sort(gaze_data_right[1,:])
        subplot_horizontal_err.plot(self.data_correction.poly_right_y(gaze_data_right_y)*self.screen_width_px, gaze_data_right_y*self.screen_height_px, color="orange", linewidth=3.0)
#        subplot_horizontal_err.set_title("Right eye horizontal error as gaze varies vertically")
        subplot_horizontal_err.set_xlabel("Horizontal error (pixels)", fontsize=18)
        subplot_horizontal_err.set_ylabel("Screen height (pixels)", fontsize=18)
        subplot_horizontal_err.set_ylim(0,self.screen_height_px)
        subplot_horizontal_err.tick_params(axis='both', which='major', labelsize=14)
        subplot_vertical_err.legend([fitname + " fit","gaze error (right eye)"])
#
        plt.show()
        
        
#        subplot_vertical_err.scatter(px_gaze_avg_x, [-e for e in px_err_avg_y]) 
#        subplot_vertical_err.plot(px_gaze_avg_x, -poly_avg_x(px_gaze_avg_x), color="orange") 
#        subplot_vertical_err.set_xlabel("Screen width (pixels)")
#        subplot_vertical_err.set_ylabel("Vertical error (pixels)")
#        subplot_vertical_err.set_xlim(0, self.screen_width_px)
#         
#        subplot_horizontal_err.scatter(px_err_avg_x, px_gaze_avg_y)  
#        subplot_horizontal_err.plot(poly_avg_y(px_gaze_avg_y), px_gaze_avg_y, color="orange")
#        subplot_horizontal_err.set_xlabel("Horizontal error (pixels)")
#        subplot_horizontal_err.set_ylabel("Screen height (pixels)")
#        subplot_horizontal_err.set_ylim(0,self.screen_height_px)
##        
#        plt.show()
        
        
        
        
    
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
    
    def get_pattern_eq(self, pathing, targets):
#        (-0.5,-0.5), (0.3, 0.5), (0.5, -0.5), (0.0, 0.0)
        equations = [];
        count = 0
        
        # filter target to turning pointpositions
        positions = targets
        if (pathing == "linear"): # iterate line segments
            prevSlope = None
            prevYInter = None
            for i in range(len(positions)):
                if i == len(positions)-1:
                    break
                p = positions[i]
                q = positions[i+1]
                
                p_x = round(p[0], 5)
                p_y = round(p[1], 5)
                q_x = round(q[0], 5)
                q_y = round(q[1], 5)
                
                if p_x == q_x and p_y == q_y:
                    continue
                slope = round((p_y-q_y)/(p_x-q_x), 5)
                y_inter = round(slope * -p_x + p_y, 5)
                slope_min = abs(slope * 0.95)
                slope_max = abs(slope * 1.05)

                if prevSlope < slope_min or slope_max < prevSlope:
                    prevSlope = abs(slope)
                    prevYInter = y_inter
                    count = 0
                    
                else:
                    count += 1
                    
                
                if count == 9:
                    equations.append((i-count, slope, y_inter))
            
        elif pathing == "circle": # calc center and avg radius to determine circ eq
            ## find opposing points to make a diamter line
#            targets = targets.T
#            avg = np.mean(targets, axis=1)
#            avg_r = np.array([((p[0]-avg[0])**2 + (p[1]-avg[1])**2)**0.5 for p in targets]).mean()
#            
#            best = targets[:,0]
#            dist = ((best[0]-avg[0])**2 + (best[1]-avg[1])**2)**0.5
#            for p in targets.T:
##                _dist = ((p[0]-avg[0])**2 + (p[1]-avg[1])**2)**0.5
##                print(_dist)
#                if abs(((p[0]-avg[0])**2 + (p[1]-avg[1])**2)**0.5 - avg_r) < abs(dist - avg_r):
#                    best = p
#                    dist = ((best[0]-avg[0])**2 + (best[1]-avg[1])**2)**0.5
#            equations.append((avg, best, dist))
#            targets = targets.T
            
            
#            diameters = []
#            for i in range(len(targets)/2):
#                p = targets[i]
#                q = targets[(i+(len(targets)-1)/2)%len(targets)]
#                d = ((p[0]-q[0])**2 + (p[1]-q[1])**2) ** 0.5
#                diameters.append(d)
#            r_avg = np.array(diameters).mean()/2
#            
#            best = None
#            for i in range(len(targets)):
#                p = targets[i]
#                q = targets[(i+(len(targets)-1)/2)%len(targets)]
#                r = ((p[0]-q[0])**2 + (p[1]-q[1])**2) ** 0.5 / 2
#                if best == None:
#                    best = (r, p, q)
#                elif abs(r_avg - r) < abs(r_avg - best[0]):
#                    best = (r, p, q)
#                    
##            print(best)
#            dx = (best[1][0]-best[2][0])/2
#            dy = (best[1][1]-best[2][1])/2
#            s = (best[2][0]+dx, best[2][1]+dy)  # (x0,y0)
#            
#            a = best[0]
#            b = abs(r_avg**2-a**2)**0.5
#            
##            a = (dx**2+dy**2)**0.5
##            b = (best[0]**2-a**2)**0.5
#            
#            x3 = s[0] + b*dy/a
#            y3 = s[1] - b*dx/a
##            print((x3, y3))
#            center_pos = (x3,y3)
#            start_pos = best[1]     # or best[2]
#            radius = best[0]
##            equations.append((center_pos, start_pos, radius))
#            
#            equations.append((center_pos, start_pos, radius))
            
            #  http://www.cs.bsu.edu/homepages/kjones/kjones/circles.pdf
            n = len(targets)
            
            sumx = sum([p[0] for p in targets])
            sumxx = sum([p[0]**2 for p in targets])
            
            sumy = sum([p[1] for p in targets])
            sumyy = sum([p[1]**2 for p in targets])
            
            d11 = n * sum([p[0] * p[1] for p in targets]) - sumx * sumy
        
            d20 = n * sumxx - sumx * sumx
            d02 = n * sumyy - sumy * sumy
        
            d30 = n * sum([p[0]**3 for p in targets]) - sumxx * sumx
            d03 = n * sum([p[1]**3 for p in targets]) - sumyy * sumy
        
            d21 = n * sum([p[0]**2 * p[1] for p in targets]) - sumxx * sumy
            d12 = n * sum([p[0] * p[1]**2 for p in targets]) - sumyy * sumx
        
            x = ((d30 + d12) * d02 - (d03 + d21) * d11) / (2 * (d20 * d02 - d11 * d11))
            y = ((d03 + d21) * d20 - (d30 + d12) * d11) / (2 * (d20 * d02 - d11 * d11))
        
            c = (sumxx + sumyy - 2 * x * sumx - 2 * y * sumy) / n
            r = (c + x**2 + y**2)**0.5
#            pdfR = sum([((p[0]-x)**2+(p[1]-y)**2)**0.5/n for p in targets])
#            print(pdfR)
            equations.append(((x,y), r))
            
        return equations

    def best_fit(self, X, Y):
    
        xbar = sum(X)/len(X)
        ybar = sum(Y)/len(Y)
        n = len(X) # or len(Y)
    
        numer = float(sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar)
        denum = float(sum([xi**2 for xi in X]) - n * xbar**2)
    
        a = numer / denum
        b = ybar - a * xbar
        
        return a, b
        
    def get_avg(self, data1, data2):
        return np.mean(np.array([data1, data2]), axis=0)
    
    def fetch_transformations(self):
        left = self.data_correction.get_trans_matrix_left()
        right = self.data_correction.get_trans_matrix_right()
        return left, right
    
    def plot_visual_angle_ring(self, center, angles_degrees):
        fig, ax = plt.subplots()              
        plt.gca().xaxis.tick_top()
        plt.xlim(0,1)
        plt.ylim(1,0)
       
        colors = ['#ff7f0e', 'purple', '#2ca02c', 'red', 'cyan', 'yellow', 'green', 'brown', 'darkgrey', 'orange', 'mediumspringgreen', 'cadetblue', 'fuchsia', 'crimson']
        
        stimuli = plt.Circle(center, 0.0375, fill=True, color='#1f77b4', linewidth=3, label="stimuli")
        ax.add_artist(stimuli)
                  
        angle_rings = [stimuli]
        for i in range(len(angles_degrees)):
            theta = angles_degrees[i]
            r = self.dist_to_screen_cm*self.ppcm*math.tan(theta*math.pi/180)
            # normalize r from pixels
            r = r / self.screen_width_px
            x = center[0] - r
            angle_ring = plt.Circle(center, r, fill=False, color=colors[i%len(colors)], linewidth=2, label=r'$\theta$ = ' + str(theta) + r'$^\circ$')
            ax.add_artist(angle_ring)
            angle_rings.append(angle_ring)
            
        ax.legend(handles=angle_rings)
        plt.show()


                    