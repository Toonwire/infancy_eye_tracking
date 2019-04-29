# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:02:06 2019

@author: Toonw
"""



import gaze_data_analyzer as gda
import numpy as np
import matplotlib.pyplot as plt


session_folder = "session_data/"
session_groups = [
    ["infant_d25_gudrun_5m","infant_d25_noel_5m"],
    ["infant_walther_2y_twin1_cp","infant_d25_viggo_2y_twin1", "infant_d25_josefine_2y", "infant_d25_molly_5y"]
]
type_of_cal = "custom_2p"
type_of_training = "fixation"

# Filtering data by
filtering_method = "dbscan_fixation"



analyzer = gda.GazeDataAnalyzer()


data_raw = []
data_cor = []
data_labels = []
colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
gaze_data = []
gaze_data_corrected = []
targets = None

subject = 1

for sessions in session_groups:
    
    data_single_raw = []
    data_single_cor = []
    
    for session in sessions:
    
        try:
            session_path = session_folder + session + "/"
            config_filename = session_path + "config.csv"
            test_path = session_path + "test_" + type_of_cal + "/"
            transformation_filename = test_path + "transformation.csv"
            training_filename = test_path + "training_" + type_of_training + ".csv"
            
            analyzer.setup(config_filename, transformation_filename, "dbscan_fixation")
            analyzer.analyze(transformation_filename, "dbscan_fixation")
            
            targets, gaze_left, gaze_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected = analyzer.analyze(training_filename, filtering_method)
            
            gaze_data.append(np.mean(np.array([gaze_left, gaze_right]), axis=0))
            gaze_data_corrected.append(np.mean(np.array([gaze_data_left_corrected, gaze_data_right_corrected]), axis=0))
    
            angle_err = np.mean(np.array([angle_err_left, angle_err_right]), axis=0)
            angle_err_corrected = np.mean(np.array([angle_err_left_corrected, angle_err_right_corrected]), axis=0)
            
            data_single_raw.append(angle_err)
            data_single_cor.append(angle_err_corrected)
            #data_labels.append(session_path.split('_')[-1])

            
        except Exception as e:
            print(e)
            
    data_session_raw = []
    data_session_cor = []
    for raw, cor in zip(data_single_raw, data_single_cor):
        data_session_raw.extend(raw)
        data_session_cor.extend(cor)
        
    data_raw.append(data_session_raw)
    data_cor.append(data_session_cor)
    data_labels.append("Group " + str(subject))
    subject += 1



#for data in data_raw:
#    plot_angle_err, = plt.plot(range(0,self.N), err_left, color='red', label="left eye")
#plt.legend(handles=[plot_left, plot_right])
#plt.title(title_string)
#plt.ylim(0,y_max)
#plt.show()
    
    
# BOXPLOT
fig = plt.figure(1, figsize=(9,12))
ax_raw = fig.add_subplot(2,1,1)
ax_cor = fig.add_subplot(2,1,2)
ax_raw.boxplot(data_raw)
ax_cor.boxplot(data_cor)
ax_raw.set_xticklabels(data_labels)
ax_cor.set_xticklabels(data_labels)

fig.show()





############################################
# Scatter plot for two charts side by side #
############################################

#data_labels.append("targets")
#
#fig = plt.figure(1, figsize=(18,6))
#ax_gaze_raw = fig.add_subplot(1,2,1)
#for idx, data in enumerate(gaze_data):
#    scatters.append(ax_gaze_raw.scatter(data[0,:], data[1,:], marker="x", color=colors[idx], alpha=0.8))
#
#scatters.append(ax_gaze_raw.scatter(targets[0,:], targets[1,:], marker="o", color="black"))
#ax_gaze_raw.legend(scatters, data_labels)
#
#ax_gaze_raw.set_title("Raw data", y=1.08)
#
#plt.gca().xaxis.tick_top()
#plt.xlim(0,1)
#plt.ylim(1,0)
#
#
#ax_gaze_cor = fig.add_subplot(1,2,2)
#for idx, data in enumerate(gaze_data_corrected):
#    scatters_corrected.append(ax_gaze_cor.scatter(data[0,:], data[1,:], marker="x", color=colors[idx], alpha=0.8))
#
#scatters_corrected.append(ax_gaze_cor.scatter(targets[0,:], targets[1,:], marker="o", color="black"))
#ax_gaze_cor.legend(scatters_corrected, data_labels)
#ax_gaze_cor.set_title("Transformed data", y=1.08)
#
#plt.gca().xaxis.tick_top()
#plt.xlim(0,1)
#plt.ylim(1,0)
#plt.show()
