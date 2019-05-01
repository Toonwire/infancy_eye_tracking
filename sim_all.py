# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda
import numpy as np

def analyze(session_folder):
    
    print("Running for " + session_folder)
    print("------------------------")
    
    # Setting path and files
    session_path = "session_data/" + session_folder + "/"
    test_folder = session_path + "test_" + type_of_cal + "/"
    config_filename = session_path + "config.csv"
    cal_filename = test_folder + "training_fixation.csv"
#    cal_filename = test_folder + "training_pursuit_circle.csv"
#    cal_filename = test_folder + "training_pursuit_linear.csv"
#    cal_filename = test_folder + "training_pursuit_spiral.csv"
    
    print("")
    print("Computing analyze linear transformation")
    print("------------------------")
    
    analyzer = gda.GazeDataAnalyzer()
    print("\nSETUP TRANSFORMATION")
    analyzer.setup_poly(config_filename, cal_filename, "dbscan_fixation")
#    analyzer.setup(config_filename, cal_filename, "dbscan_pursuit")
#    print("\nTRAINING DATA")
#    analyzer.analyze(cal_filename, "dbscan_fixation")
    
    print("\nTEST DATA - FIXATION")
    training_filename = test_folder + "training_fixation.csv"
    rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = analyzer.analyze_poly(training_filename, "dbscan_fixation", "values")

    fixation_deg_raw.append(rmse_deg_raw)    
    fixation_deg_cor.append(rmse_deg_cor)
    fixation_deg_imp.append(rmse_deg_imp)
    
    print("\nTEST DATA - PURSUIT (CIRCLE)")
    training_filename = test_folder + "training_pursuit_circle.csv"
    rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = analyzer.analyze_poly(training_filename, "dbscan_pursuit", "values")

    pursuit_circle_deg_raw.append(rmse_deg_raw)    
    pursuit_circle_deg_cor.append(rmse_deg_cor)
    pursuit_circle_deg_imp.append(rmse_deg_imp)
    
    print("\nTEST DATA - PURSUIT (LINEAR)")
    training_filename = test_folder + "training_pursuit_linear.csv"
    rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = analyzer.analyze_poly(training_filename, "dbscan_pursuit", "values")

    pursuit_linear_deg_raw.append(rmse_deg_raw)
    pursuit_linear_deg_cor.append(rmse_deg_cor)
    pursuit_linear_deg_imp.append(rmse_deg_imp)

    print("\nTEST DATA - PURSUIT (SPIRAL)")
    training_filename = test_folder + "training_pursuit_spiral.csv"
    rmse_deg_raw, rmse_deg_cor, rmse_deg_imp = analyzer.analyze_poly(training_filename, "dbscan_pursuit", "values")

    pursuit_spiral_deg_raw.append(rmse_deg_raw)
    pursuit_spiral_deg_cor.append(rmse_deg_cor)
    pursuit_spiral_deg_imp.append(rmse_deg_imp)


    '''
    print("") 
    print("Computing analyze linear transformation mix")
    print("------------------------")

    
    analyzer = gda.GazeDataAnalyzer()
    print("\nSETUP TRANSFORMATION")
    analyzer.setup_seb(config_filename, cal_filename, "dbscan_fixation")
    print("\nTRAINING DATA")
    analyzer.analyze_seb(cal_filename, "dbscan_fixation")
    print("\nTEST DATA - FIXATION")
    training_filename = test_folder + "training_fixation.csv"
    analyzer.analyze_seb(training_filename, "dbscan_fixation")
    print("\nTEST DATA - PURSUIT (CIRCLE)")
    training_filename = test_folder + "training_pursuit_circle.csv"
    analyzer.analyze_seb(training_filename, "dbscan_pursuit")
    print("\nTEST DATA - PURSUIT (LINEAR)")
    training_filename = test_folder + "training_pursuit_linear.csv"
    analyzer.analyze_seb(training_filename, "dbscan_pursuit")
    print("\nTEST DATA - PURSUIT (SPIRAL)")    
    training_filename = test_folder + "training_pursuit_spiral.csv"
    analyzer.analyze_seb(training_filename, "dbscan_pursuit")


    print("")
    print("Computing analyze regression by data driven")
    print("------------------------")

    analyzer = gda.GazeDataAnalyzer()
    print("\nSETUP TRANSFORMATION")
    analyzer.setup_regression(config_filename, cal_filename, "dbscan_fixation")
    print("\nTRAINING DATA")
    analyzer.analyze_regression(cal_filename, "dbscan_fixation")
    print("\nTEST DATA - FIXATION")
    training_filename = test_folder + "training_fixation.csv"
    analyzer.analyze_regression(training_filename, "dbscan_fixation")
    print("\nTEST DATA - PURSUIT (CIRCLE)")
    training_filename = test_folder + "training_pursuit_circle.csv"
    analyzer.analyze_regression(training_filename, "dbscan_pursuit")
    print("\nTEST DATA - PURSUIT (LINEAR)")
    training_filename = test_folder + "training_pursuit_linear.csv"
    analyzer.analyze_regression(training_filename, "dbscan_pursuit")
    print("\nTEST DATA - PURSUIT (SPIRAL)")    
    training_filename = test_folder + "training_pursuit_spiral.csv"
    analyzer.analyze_regression(training_filename, "dbscan_pursuit")

    print("")
    
    print("Computing analyze regression by optimization")
    print("------------------------")

    analyzer = gda.GazeDataAnalyzer()
    print("\nSETUP TRANSFORMATION")
    analyzer.setup_poly(config_filename, cal_filename, "dbscan_fixation")
    print("\nTRAINING DATA")
    analyzer.analyze_poly(cal_filename, "dbscan_fixation")
    print("\nTEST DATA - FIXATION")
    training_filename = test_folder + "training_fixation.csv"
    analyzer.analyze_poly(training_filename, "dbscan_fixation")
    print("\nTEST DATA - PURSUIT (CIRCLE)")
    training_filename = test_folder + "training_pursuit_circle.csv"
    analyzer.analyze_poly(training_filename, "dbscan_pursuit")
    print("\nTEST DATA - PURSUIT (LINEAR)")
    training_filename = test_folder + "training_pursuit_linear.csv"
    analyzer.analyze_poly(training_filename, "dbscan_pursuit")
    print("\nTEST DATA - PURSUIT (SPIRAL)")    
    training_filename = test_folder + "training_pursuit_spiral.csv"
    analyzer.analyze_poly(training_filename, "dbscan_pursuit")
    '''
    
    print("")
    
    
print("Calibrating for default:")
print("------------------------")
# Run analyse on
type_of_cal = "default"

print("")

fixation_deg_raw = []
fixation_deg_cor = []
fixation_deg_imp = []
pursuit_circle_deg_raw = []
pursuit_circle_deg_cor = []
pursuit_circle_deg_imp = []
pursuit_linear_deg_raw = []
pursuit_linear_deg_cor = []
pursuit_linear_deg_imp = []
pursuit_spiral_deg_raw = []
pursuit_spiral_deg_cor = []
pursuit_spiral_deg_imp = []

# Session to run
analyze("ctrl_group_2_louise")
analyze("ctrl_group_2_lasse")
analyze("ctrl_group_2_marie")
analyze("ctrl_group_2_mikkel")
analyze("ctrl_group_2_lukas")
analyze("ctrl_group_2_seb")

def print_nice(values):
    my_str = ""
    for i in range(len(values)):
        my_str += str(values[i]) + "\t"
    print(my_str)

print("Improvements values")
print_nice(fixation_deg_imp)
print_nice(pursuit_circle_deg_imp)
print_nice(pursuit_linear_deg_imp)
print_nice(pursuit_spiral_deg_imp)

print("Correction values")
print_nice(fixation_deg_cor)
print_nice(pursuit_circle_deg_cor)
print_nice(pursuit_linear_deg_cor)
print_nice(pursuit_spiral_deg_cor)

print("Raw values")
print_nice(fixation_deg_raw)
print_nice(pursuit_circle_deg_raw)
print_nice(pursuit_linear_deg_raw)
print_nice(pursuit_spiral_deg_raw)


#
#print("Fixations value")
#print(fixation_deg_imp)
#print(fixation_deg_cor)
#
#print("Pursuit circle value")
#print(pursuit_circle_deg_imp)
#print(pursuit_circle_deg_cor)
#
#print("Pursuit linear value")
#print(pursuit_linear_deg_imp)
#print(pursuit_linear_deg_cor)
#
#print("Pursuit spiral value")
#print(pursuit_spiral_deg_imp)
#print(pursuit_spiral_deg_cor)


print("")
print("Fixations value")
print("Average RMSE degree after correction: " + str(np.mean(fixation_deg_cor)))
print("Average RMSE improvement: " + str(np.mean(fixation_deg_imp)))

print("")
print("Pursuit circle value")
print("Average RMSE degree after correction: " + str(np.mean(pursuit_circle_deg_cor)))
print("Average RMSE improvement: " + str(np.mean(pursuit_circle_deg_imp)))

print("")
print("Pursuit linear value")
print("Average RMSE degree after correction: " + str(np.mean(pursuit_linear_deg_cor)))
print("Average RMSE improvement: " + str(np.mean(pursuit_linear_deg_imp)))

print("")
print("Pursuit spiral value")
print("Average RMSE degree after correction: " + str(np.mean(pursuit_spiral_deg_cor)))
print("Average RMSE improvement: " + str(np.mean(pursuit_spiral_deg_imp)))