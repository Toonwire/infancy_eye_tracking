# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:02:06 2019

@author: Toonw
"""



import gaze_data_analyzer as gda
import numpy as np
import matplotlib.pyplot as plt


session_folder = "session_data/"
#sessions = ["ctrl_group_chrille1", "ctrl_group_lasse", "ctrl_group_louise", "ctrl_group_marie", "ctrl_group_mikkel"]
#sessions = ["ctrl_group_chrille1"]
#sessions = ["ctrl_group_lasse"]
#sessions = ["ctrl_group_louise", "ctrl_group_mikkel"]
#sessions = ["ctrl4_a_seb_glass","ctrl4_a_seb","ctrl4_a_marie_2","ctrl4_a_marie","ctrl4_a_lukas_blind","ctrl4_a_lukas"]
#sessions = ["infant_d25_gudrun_5m","infant_d25_noel_5m"]
#sessions = ["infant_walther_2y_twin1_cp","infant_d25_viggo_2y_twin1", "infant_d25_josefine_2y", "infant_d25_molly_5y"]
#sessions = ["ctrl_group_louise"]
sessions = ["infant_d25_noel_5m","infant_d25_gudrun_5m","infant1_d2_viggo_6m","infant1_d52_vilja_7m"]
#sessions = ["infant_walther_2y_twin1_cp","infant_d25_viggo_2y_twin1"]

types_of_cal = ["custom_2p", "custom_5p", "default"]
name_of_cal = ["2-Point", "5-Point", "Default + Correction"]

#types_of_cal = ["active","custom_2p", "custom_5p", "default"]
#name_of_cal = ["Tobii", "2-Point", "5-Point", "Default + Correction"]


type_of_training = "fixation"
#type_of_training = "pursuit_linear"
#type_of_training = "pursuit_spiral"

# Filtering data by
#filtering_method = None
filtering_method = "dbscan_fixation"
#filtering_method = "dbscan_pursuit"
#filtering_method = "threshold_time_pursuit"

type_of_training_2 = "pursuit_linear"
filtering_method_2 = "dbscan_pursuit"

remove_outliers = False

analyzer = gda.GazeDataAnalyzer()


data_raw = []
data_cor = []
data_labels = []
colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'brown', 'darkgrey', 'orange', 'mediumspringgreen', 'cadetblue', 'fuchsia', 'crimson']
                
gaze_data = []
gaze_data_corrected = []
all_targets = [[],[]]



for cal_type, name in zip(types_of_cal,name_of_cal):

    result_angle = np.array([])
    result_angle_cor = np.array([])

    for session in sessions:
        
        try:
        
            session_path = session_folder + session + "/"
            config_filename = session_path + "config.csv"
            test_path = session_path + "test_" + cal_type + "/"
            transformation_filename = test_path + "transformation.csv"
            training_filename = test_path + "training_" + type_of_training + ".csv"
            
            analyzer.setup_affine2(config_filename, transformation_filename, "dbscan_fixation")
            #analyzer.analyze(transformation_filename, "dbscan_fixation")
            
            targets, gaze_left, gaze_right, gaze_data_left_corrected, gaze_data_right_corrected, angle_err_left, angle_err_right, angle_err_left_corrected, angle_err_right_corrected = analyzer.analyze_affine2(training_filename, filtering_method, remove_outliers = remove_outliers)
            
            gaze_data.append(np.mean(np.array([gaze_left, gaze_right]), axis=0))
            gaze_data_corrected.append(np.mean(np.array([gaze_data_left_corrected, gaze_data_right_corrected]), axis=0))
            
            for t in targets.T:
                all_targets[0].append(t[0])
                all_targets[1].append(t[1])
    
            angle_err = np.mean(np.array([angle_err_left, angle_err_right]), axis=0)
            angle_err_corrected = np.mean(np.array([angle_err_left_corrected, angle_err_right_corrected]), axis=0)                
        
            result_angle = np.concatenate((result_angle, angle_err))
            result_angle_cor = np.concatenate((result_angle_cor, angle_err_corrected))
        
            if not type_of_training_2 == None:
                training_filename_2 = test_path + "training_" + type_of_training_2 + ".csv"
                targets_2, gaze_left_2, gaze_right_2, gaze_data_left_corrected_2, gaze_data_right_corrected_2, angle_err_left_2, angle_err_right_2, angle_err_left_corrected_2, angle_err_right_corrected_2 = analyzer.analyze_affine2(training_filename_2, filtering_method_2, remove_outliers = remove_outliers)
    
                angle_err_2 = np.mean(np.array([angle_err_left_2, angle_err_right_2]), axis=0)
                angle_err_corrected_2 = np.mean(np.array([angle_err_left_corrected_2, angle_err_right_corrected_2]), axis=0)
    
                result_angle = np.concatenate((result_angle, angle_err_2))
                result_angle_cor = np.concatenate((result_angle_cor, angle_err_corrected_2))
        
        except Exception as e:
            print(e)

    data_raw.append(result_angle)
    data_cor.append(result_angle_cor)
    
    #data_labels.append(session_path.split('_')[-1])
    data_labels.append(name)
#    
        
all_targets = np.array(all_targets)

data_labels.append("targets")

#scatters = []
#for idx, data in enumerate(gaze_data):
#    scatters.append(plt.scatter(data[0,:], data[1,:], marker="x", color=colors[idx], alpha=0.8))    
#
#scatters.append(plt.scatter(all_targets[0,:], all_targets[1,:], marker="^", color="black"))
##plt.legend(scatters, data_labels)
#plt.title("Raw data", y=1.08)
#plt.gca().xaxis.tick_top()
#
#plt.xlim(0,1)
#plt.ylim(1,0)

#ax = plt.gca()
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#plt.legend(scatters, data_labels, loc="center left", bbox_to_anchor=(1,0.5))

#plt.show()
#
#scatters_corrected = []
#for idx, data_corrected in enumerate(gaze_data_corrected):
#    scatters_corrected.append(plt.scatter(data_corrected[0,:], data_corrected[1,:], marker="x", color=colors[idx], alpha=0.8))
#
#scatters_corrected.append(plt.scatter(all_targets[0,:], all_targets[1,:], marker="^", color="black"))
##plt.legend(scatters_corrected, data_labels)
#plt.title("Transformed data", y=1.08)
#plt.gca().xaxis.tick_top()
#plt.xlim(0,1)
#plt.ylim(1,0)
#plt.show()
#


#for data in data_raw:
#    plot_angle_err, = plt.plot(range(0,self.N), err_left, color='red', label="left eye")
#plt.legend(handles=[plot_left, plot_right])
#plt.title(title_string)
#plt.ylim(0,y_max)
#plt.show()
    
    
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

# BOXPLOT
fig = plt.figure(1, figsize=(9,12))
ax_raw = fig.add_subplot(2,1,1)
ax_cor = fig.add_subplot(2,1,2)
ax_raw.boxplot(data_raw)
ax_cor.boxplot(data_cor)
ax_raw.set_xticklabels(data_labels)
ax_cor.set_xticklabels(data_labels)

ax_raw.set_ylim(0,11)
ax_cor.set_ylim(0,11)

fig.show()


raw_data = []
for row in data_raw:
    raw_data.extend(row)
    print("Raw")
    print("STD: " + str(np.std(row)))
    print("AVG: " + str(np.average(row)))
    print("1Q: " + str(np.percentile(row, 25)))
    print("2Q: " + str(np.percentile(row, 50)))
    print("3Q: " + str(np.percentile(row, 75)))

    print("")
    
    
cor_data = []
for row in data_cor:
    cor_data.extend(row)
    print("Corrected")
    print("STD: " + str(np.std(row)))
    print("AVG: " + str(np.average(row)))
    print("1Q: " + str(np.percentile(row, 25)))
    print("2Q: " + str(np.percentile(row, 50)))
    print("3Q: " + str(np.percentile(row, 75)))
    print("")
    
print("Raw")
print("STD: " + str(np.std(np.array(raw_data))))
print("AVG: " + str(np.average(np.array(raw_data))))
print("")
print("Corrected")
print("STD: " + str(np.std(np.array(cor_data))))
print("AVG: " + str(np.average(np.array(cor_data))))

print("")

print("Raw")
print("1Q: " + str(np.percentile(np.array(raw_data), 25)))
print("2Q: " + str(np.percentile(np.array(raw_data), 50)))
print("3Q: " + str(np.percentile(np.array(raw_data), 75)))
print("")
print("Corrected")
print("1Q: " + str(np.percentile(np.array(cor_data), 25)))
print("2Q: " + str(np.percentile(np.array(cor_data), 50)))
print("3Q: " + str(np.percentile(np.array(cor_data), 75)))




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
