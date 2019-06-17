# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:21:30 2019

@author: Toonw
"""


import gaze_data_analyzer as gda
import numpy as np


# Run analyse on
type_of_cal = "default"
#type_of_cal = "custom_2p"
#type_of_cal = "custom_5p"


#session_folders = ["ctrl_group_2_lasse", "ctrl_group_3_lukas", "ctrl_group_3_seb", 
#                   "ctrl_group_2_marie", "ctrl_group_chrille2"]

session_folders = ["infant2_d52_vilja_7m", "infant3_d_marley_7m_2", "infant2_d2_viggo_6m",
                   "infant_d25_noel_5m", "infant_d25_gudrun_5m" , "infant_d25_josefine_4m"]

session_folders = ["ctrl_group_3_seb"]
analyzer = gda.GazeDataAnalyzer()


deg_raw = []
deg_corrected = []
deg_corrected_mean_replace = []
for i in range(len(session_folders)):
    session_path = "session_data/" + session_folders[i] + "/"
    test_folder = session_path + "test_" + type_of_cal + "/"
    config_filename = session_path + "config.csv"
    cal_filename = test_folder + "training_fixation.csv"
    analyzer.setup(config_filename, cal_filename, "dbscan_fixation")    
    training_filename = test_folder + "training_pursuit_circle.csv"
    
    rmse_deg_raw, rmse_deg_cor,_ = analyzer.analyze(training_filename, "dbscan_pursuit", output="values", mean_cluster_replacement=False)
    _,rmse_deg_cor_mean,_ = analyzer.analyze(training_filename, "dbscan_pursuit", output="values", mean_cluster_replacement=True)
    deg_corrected.append(rmse_deg_cor)
    deg_corrected_mean_replace.append(rmse_deg_cor_mean)
    deg_raw.append(rmse_deg_raw)

print("Raw RMSE")
print(deg_raw)
print("Avg RMSE = " + str(np.array(deg_raw).mean()))
print("")
print("Standard correction RMSE")
print(deg_corrected)
print("Avg RMSE = " + str(np.array(deg_corrected).mean()))
print("")
print("Mean cluster replacement correction RMSE")
print(deg_corrected_mean_replace)
print("Avg RMSE = " + str(np.array(deg_corrected_mean_replace).mean()))
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    