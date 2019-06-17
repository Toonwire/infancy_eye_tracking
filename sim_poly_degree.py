# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda
import numpy as np



# Run analyse on
type_of_cal = "default"
#type_of_cal = "custom_2p"
#type_of_cal = "custom_5p"


session_folders = ["ctrl_group_2_lasse", "ctrl_group_3_lukas", "ctrl_group_3_seb", 
                   "ctrl_group_2_marie", "ctrl_group_chrille2", "ctrl_group_2_louise"]

#session_folders = ["infant2_d52_vilja_7m", "infant3_d_marley_7m_2", 
#                   "infant_d25_noel_5m", "infant_d25_viggo_2y_twin1", "infant_d25_josefine_4m"]

session_folders = ["ctrl_group_2_louise"]
analyzer = gda.GazeDataAnalyzer()

max_poly_degree = 6
all_rmse = []
for i in range(len(session_folders)):
    # Setting path and files
    session_path = "session_data/" + session_folders[i] + "/"
    test_folder = session_path + "test_" + type_of_cal + "/"
    config_filename = session_path + "config.csv"
    cal_filename = test_folder + "training_pursuit_linear.csv"
    training_filename = test_folder + "training_pursuit_linear.csv"

    rmse = []
    for j in range(max_poly_degree):
        print("POLYNOMIAL DEGREE: " + str(j+1))
        analyzer.setup_regression(config_filename, cal_filename, "dbscan_pursuit", poly_degree=j+1)
        rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = analyzer.analyze_regression(training_filename, "dbscan_pursuit", output="values")
        rmse.append(rmse_deg_cor)
    all_rmse.append(rmse)
    
    print(session_folders[i])
    for j in range(len(rmse)):
        print("Degree " + str(j+1) + ":\t rmse = "+str(rmse[j]))
    print("")

all_rmse = np.array(all_rmse)
print("All RMSE avg across participant\n--------------------")
for i in range(len(all_rmse[0,:])):
    avg = np.average(all_rmse[:,i])
    print("Degree " + str(i+1) + ":\t rmse(avg) = "+str(avg))
print("")
    
import matplotlib.pyplot as plt

print(all_rmse)

data_labels = ["1. degree", "2. degree", "3. degree", "4. degree", "5. degree", "6. degree"]
fig = plt.figure(1, figsize=(9,12))
ax = fig.add_subplot(2,1,1)
ax.boxplot(all_rmse)
ax.set_xticklabels(data_labels, fontsize=14)
ax.set_yticklabels([x for x in range(8)], fontsize=14)
ax.set_ylim(0,7)
ax.set_ylabel("RMSE of Visual Angle", fontsize=18)
ax.set_xlabel("Polynomial Degree", fontsize=18)

fig.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    