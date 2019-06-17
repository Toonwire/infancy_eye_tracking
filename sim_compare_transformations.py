# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:35:03 2019

@author: Toonw
"""

import gaze_data_analyzer as gda
import numpy as np
import matplotlib.pyplot as plt



# Run analyse on
#type_of_cal = "default"
type_of_cal = "custom_2p"
#type_of_cal = "custom_5p"

# Session to run

#session_folder = "infant2_525d_noel_6m"
#session_folder = "infant2_d52_vilja_7m"
session_folder = "infant_d25_noel_5m"
#session_folder = "linear_special_2p_seb"

session_folders = ["infant2_525d_noel_6m", "infant2_d52_vilja_7m", "ctrl_group_mikkel", 
                   "ctrl_group_3_lukas", "ctrl_group_3_seb", 
                   "infant2_52d_sofie_7m", "infant2_d2_viggo_6m", "infant_d25_gudrun_5m"]


alphas = []

aa = []
bb = []
cc = []
dd = []

for i in range(len(session_folders)):
    # Setting path and files
    session_path = "session_data/" + session_folders[i] + "/"
    test_folder = session_path + "test_" + type_of_cal + "/"
    config_filename = session_path + "config.csv"
    cal_filename = test_folder + "training_pursuit_linear.csv"
    
    analyzer = gda.GazeDataAnalyzer()
    
    print("\nSETUP TRANSFORMATION")
    analyzer.setup(config_filename, cal_filename, "dbscan_pursuit")
    
    T_left, T_right = analyzer.fetch_transformations()
#    A = [[a,b], [c,d]]
#    T_left = A.T_right
#    a*T_right[0,0] + b*T_right[1,0] = T_left[0,0]
#    a*T_right[1,0] + b*T_right[1,1] = T_left[0,1]
#    c*T_right[0,0] + d*T_right[1,0] = T_left[1,0]
#    c*T_right[1,0] + d*T_right[1,1] = T_left[1,1]
#    two x two equations with two unknowns
#    ------------
    
    l1 = T_left[0,0]
    l2 = T_left[0,1]
    l3 = T_left[1,0]
    l4 = T_left[1,1]
    
    r1 = T_right[0,0]
    r2 = T_right[0,1]
    r3 = T_right[1,0]
    r4 = T_right[1,1]
    
    
    a = (l2*r3-l1*r4)/(r2*r3-r1*r4) 
    b = (l2*r1-l1*r2)/(r1*r4-r2*r3)
    c = (l4*r3-l1*r4)/(r2*r3-r1*r4)
    d = (l4*r1-l1*r2)/(r1*r4-r2*r3)
    
    alpha = np.array([[a,b],[c,d]])
#    print(np.matmul(alpha,T_right))
    alphas.append(alpha)
    
    aa.append(("a"+str(i), a))
    bb.append(("b"+str(i), b))
    cc.append(("c"+str(i), c))
    dd.append(("d"+str(i), d))

# Create bars
abcd = []
abcd.extend(aa)
abcd.append(("",0))
abcd.extend(bb)
abcd.append(("",0))
abcd.extend(cc)
abcd.append(("",0))
abcd.extend(dd)

y_pos = np.arange(len(abcd))

colors = ['black', 'red', 'blue', 'cyan', 'yellow', 'purple', 'green', 'brown', 'darkgrey', 'orange', 'mediumspringgreen', 'cadetblue', 'fuchsia', 'crimson']

plt.figure(figsize=(14, 3))  # width:20, height:3
plt.bar(y_pos, [v[1] for v in abcd], color=colors[0:len(session_folders)+1], align='edge', width=0.3)
 
# Create names on the x-axis
plt.xticks(y_pos,  [v[0] for v in abcd])
 
# Show graphic
plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
