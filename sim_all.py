# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:34:34 2019

@author: Toonw
"""

import gaze_data_analyzer as gda


def analyze(session_folder):
    
    print("Running for " + session_folder)
    print("------------------------")
    
    # Setting path and files
    session_path = "session_data/" + session_folder + "/"
    test_folder = session_path + "test_" + type_of_cal + "/"
    config_filename = session_path + "config.csv"
    cal_filename = test_folder + "training_fixation.csv"

    print("")
    print("Computing analyze linear transformation")
    print("------------------------")
    
    analyzer = gda.GazeDataAnalyzer()
    print("\nSETUP TRANSFORMATION")
    analyzer.setup(config_filename, cal_filename, "dbscan_fixation")
    print("\nTRAINING DATA")
    analyzer.analyze(cal_filename, "dbscan_fixation")
    print("\nTEST DATA - FIXATION")
    training_filename = test_folder + "training_fixation.csv"
    analyzer.analyze(training_filename, "dbscan_fixation")
    print("\nTEST DATA - PURSUIT (CIRCLE)")
    training_filename = test_folder + "training_pursuit_circle.csv"
    analyzer.analyze_poly(training_filename, "dbscan_pursuit")
    print("\nTEST DATA - PURSUIT (LINEAR)")
    training_filename = test_folder + "training_pursuit_linear.csv"
    analyzer.analyze(training_filename, "dbscan_pursuit")
    print("\nTEST DATA - PURSUIT (SPIRAL)")
    training_filename = test_folder + "training_pursuit_spiral.csv"
    analyzer.analyze(training_filename, "dbscan_pursuit")

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
    analyzer.analyze_poly(training_filename, "dbscan_pursuit")
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
    analyzer.analyze_poly(training_filename, "dbscan_pursuit")
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
    
    print("")
    
print("Calibrating for default:")
print("------------------------")
# Run analyse on
type_of_cal = "default"

print("")

# Session to run
analyze("ctrl_group_louise")
analyze("ctrl_group_lasse")
analyze("ctrl_group_marie")
analyze("ctrl_group_mikkel")