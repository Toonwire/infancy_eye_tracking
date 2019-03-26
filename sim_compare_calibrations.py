# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda
import numpy as np
import matplotlib.pyplot as plt


# Run analyse on
type_of_cal = ["custom_2p", "custom_5p", "custom_5p_img", "default", "active"]
type_of_training = "fixation"

# Filtering data by
filtering_method = "dbscan_fixation"

# Session to run
session_folder = "ctrl_group_lasse-kopi"


# Setting path and files
session_path = "session_data/" + session_folder + "/"
config_filename = session_path + "config.csv"    

analyzer = gda.GazeDataAnalyzer()

data_raw = []
data_cor = []
data_labels = []
colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
gaze_data = []
gaze_data_corrected = []
targets = None

for cal in type_of_cal:
    
    print("Start setting up data for " + cal)
    
    try:
        test_path = session_path + "test_" + cal + "/"
        transformation_filename = test_path + "transformation.csv"
        training_filename = test_path + "training_" + type_of_training + ".csv"
    
    
        analyzer.setup_seb(config_filename, transformation_filename, "dbscan_fixation")
        analyzer.analyze_seb(transformation_filename, "dbscan_fixation")
        
        targets, gaze_left, gaze_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected = analyzer.analyze_seb(training_filename, filtering_method)
        
        gaze_data.append(np.mean(np.array([gaze_left, gaze_right]), axis=0))
        gaze_data_corrected.append(np.mean(np.array([gaze_data_left_corrected, gaze_data_right_corrected]), axis=0))
    
        angle_err = np.mean(np.array([angle_err_left, angle_err_right]), axis=0)
        angle_err_corrected = np.mean(np.array([angle_err_left_corrected, angle_err_right_corrected]), axis=0)
        
        data_raw.append(angle_err)
        data_cor.append(angle_err_corrected)
        data_labels.append("Calibration " + cal)
    except Exception as e:
        print("Failed while setting up data for " + cal)
        print(e)
    
data_labels.append("targets")

scatters = []
for idx, data in enumerate(gaze_data):
    scatters.append(plt.scatter(data[0,:], data[1,:], marker="x", color=colors[idx], alpha=0.8))    

scatters.append(plt.scatter(targets[0,:], targets[1,:], marker="o", color="black"))
plt.legend(scatters, data_labels)
plt.title("Raw data", y=1.08)
plt.gca().xaxis.tick_top()
plt.xlim(0,1)
plt.ylim(1,0)
plt.show()

scatters_corrected = []
for idx, data_corrected in enumerate(gaze_data_corrected):
    scatters_corrected.append(plt.scatter(data_corrected[0,:], data_corrected[1,:], marker="x", color=colors[idx], alpha=0.8))

scatters_corrected.append(plt.scatter(targets[0,:], targets[1,:], marker="o", color="black"))
plt.legend(scatters_corrected, data_labels)
plt.title("Transformed data", y=1.08)
plt.gca().xaxis.tick_top()
plt.xlim(0,1)
plt.ylim(1,0)
plt.show()



# BOXPLOT
fig = plt.figure(1, figsize=(9,12))
ax_raw = fig.add_subplot(2,1,1)
ax_cor = fig.add_subplot(2,1,2)
ax_raw.boxplot(data_raw)
ax_cor.boxplot(data_cor)
ax_raw.set_xticklabels(data_labels)
ax_cor.set_xticklabels(data_labels)

fig.show()


