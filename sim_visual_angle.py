# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:21:04 2019

@author: Toonw
"""

import numpy as np
import gaze_data_analyzer as gda
import matplotlib.pyplot as plt
import math





# Run analyse on
type_of_cal = "default"
#type_of_cal = "custom_2p"
#type_of_cal = "custom_5p"

# Session to run

#session_folder = "infant2_525d_noel_6m"
#session_folder = "infant2_d52_vilja_7m"
#session_folder = "infant3_d_marley_7m_2"
session_folder = "ctrl_group_3_seb"


# Setting path and files
session_path = "session_data/" + session_folder + "/"
test_folder = session_path + "test_" + type_of_cal + "/"
config_filename = session_path + "config.csv"
cal_filename = test_folder + "training_fixation.csv"

analyzer = gda.GazeDataAnalyzer()

print("\nSETUP TRANSFORMATION")
analyzer.setup(config_filename, cal_filename, "dbscan_fixation")
#
#print("\nTEST DATA - FIXATION")
#training_filename = test_folder + "training_fixation.csv"
#analyzer.analyze(training_filename, "dbscan_fixation")

#print("\nTEST DATA - PURSUIT (CIRCLE)")
#training_filename = test_folder + "training_pursuit_circle.csv"
#analyzer.analyze(training_filename, "dbscan_pursuit")

#print("\nTEST DATA - PURSUIT (LINEAR)")
#training_filename = test_folder + "training_pursuit_linear.csv"
#_,rmse_cor,_ = analyzer.analyze(training_filename, "dbscan_pursuit",output="values")

#print("\nTEST DATA - PURSUIT (SPIRAL)")
#training_filename = test_folder + "training_pursuit_spiral.csv"
#analyzer.analyze(training_filename, "dbscan_pursuit")



analyzer.plot_visual_angle_ring(center=(0.5,0.5), angles_degrees=[1.5, 5, 10, 20])



#angle_circle = plt.Circle((0.5,0.5), 0.2, fill=False, color='orange', linewidth=3)
#plt.gcf().gca().add_artist(angle_circle)
#plt.show()




















