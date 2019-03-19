# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:02:06 2019

@author: Toonw
"""



import gaze_data_analyzer as gda
import glob
import numpy as np
import matplotlib.pyplot as plt

sessions_folder = "session_data/*"
type_of_cal = "custom_2p"
type_of_training = "pursuit"


analyzer = gda.GazeDataAnalyzer()


data_raw = []
data_cor = []
data_labels = []
colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
gaze_data = []
gaze_data_corrected = []
targets = None

subject = 1

for session_path in glob.glob(sessions_folder):
    
    try:

        config_filename = session_path + "/config.csv"    
        cal_filename = session_path + "/calibrations/cal_" + type_of_cal + ".csv"
        
        analyzer.setup(config_filename, cal_filename, "dbscan")
        analyzer.analyze(cal_filename, "dbscan")
        
        
        training_filename = session_path + "/training_with_cal_" + type_of_cal + "/training_" + type_of_training + ".csv"
        _,_,targets = analyzer.read_data(training_filename, "dbscan")
        gaze_left, gaze_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected = analyzer.analyze(training_filename, "dbscan")
        
        gaze_data.append(np.mean(np.array([gaze_left, gaze_right]), axis=0))
        gaze_data_corrected.append(np.mean(np.array([gaze_data_left_corrected, gaze_data_right_corrected]), axis=0))

        angle_err = np.mean(np.array([angle_err_left, angle_err_right]), axis=0)
        angle_err_corrected = np.mean(np.array([angle_err_left_corrected, angle_err_right_corrected]), axis=0)
        
        data_raw.append(angle_err)
        data_cor.append(angle_err_corrected)
        #data_labels.append(session_path.split('_')[-1])
        data_labels.append("Subject " + str(subject))
        
        subject += 1
        
    except Exception as e:
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
