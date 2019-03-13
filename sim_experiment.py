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
type_of_cal = "custom_5p"
type_of_training = "pursuit"


analyzer = gda.GazeDataAnalyzer()


data_raw = []
data_cor = []
data_labels = []
colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
gaze_data = []
targets = None

for session_path in glob.glob(sessions_folder):
    
    try:

        config_filename = session_path + "/config.csv"    
        cal_filename = session_path + "/calibrations/cal_" + type_of_cal + ".csv"
        
        analyzer.setup(config_filename, cal_filename)
        
        
        
        training_filename = session_path + "/training_with_cal_" + type_of_cal + "/training_" + type_of_training + ".csv"
        _,_,targets = analyzer.read_data(training_filename)
        gaze_left, gaze_right, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected = analyzer.analyze(training_filename)
        
        angle_err = np.mean(np.array([angle_err_left, angle_err_right]), axis=0)
        angle_err_corrected = np.mean(np.array([angle_err_left_corrected, angle_err_right_corrected]), axis=0)
        
        gaze_data.append(np.mean(np.array([gaze_left, gaze_right]), axis=0))
        
        data_raw.append(angle_err)
        data_cor.append(angle_err_corrected)
        data_labels.append(session_path.split('_')[-1])
        
    except:
        pass
    

plt.figure()
scatters = []

for idx, data in enumerate(gaze_data):
    scatters.append(plt.scatter(data[0,:], data[1,:], marker="x", color=colors[idx], alpha=0.8))
    
scatters.append(plt.scatter(targets[0,:], targets[1,:], marker="o", color="black"))
data_labels.append("targets")
plt.legend(scatters, data_labels)
plt.title("pursuit", y=1.08)
plt.gca().xaxis.tick_top()
plt.xlim(0,1)
plt.ylim(1,0)
plt.show()



for data in data_raw:
    plot_angle_err, = plt.plot(range(0,self.N), err_left, color='red', label="left eye")
plt.legend(handles=[plot_left, plot_right])
plt.title(title_string)
plt.ylim(0,y_max)
plt.show()
    
    
# BOXPLOT
#fig = plt.figure(1, figsize=(9,12))
#ax_raw = fig.add_subplot(2,1,1)
#ax_cor = fig.add_subplot(2,1,2)
#ax_raw.boxplot(data_raw)
#ax_cor.boxplot(data_cor)
#ax_raw.set_xticklabels(data_labels)
#ax_cor.set_xticklabels(data_labels)
#fig.show()
